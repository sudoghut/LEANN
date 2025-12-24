"""
Build a LEANN RAG index from files under ./knowledge-base (default).
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any

from leann.api import LeannBuilder
from leann.registry import register_project_directory
from llama_index.core import SimpleDirectoryReader

# Add apps directory to path so we can import chunking when running as a script.
sys.path.insert(0, str(Path(__file__).parent))
from chunking import create_text_chunks

# Optional import: older PyPI builds may not include settings helpers.
try:
    from leann.settings import resolve_ollama_host, resolve_openai_api_key, resolve_openai_base_url
except ImportError:
    import os

    def resolve_ollama_host(value: str | None) -> str | None:
        return value or os.getenv("LEANN_OLLAMA_HOST") or os.getenv("OLLAMA_HOST")

    def resolve_openai_api_key(value: str | None) -> str | None:
        return value or os.getenv("OPENAI_API_KEY")

    def resolve_openai_base_url(value: str | None) -> str | None:
        return value or os.getenv("OPENAI_BASE_URL")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed docs folder into a LEANN index.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="knowledge-base",
        help="Directory containing documents to index (default: knowledge-base)",
    )
    parser.add_argument(
        "--file-types",
        nargs="+",
        default=None,
        help="Filter by file types (e.g., .pdf .txt .md). If not set, all supported types are processed",
    )
    parser.add_argument("--chunk-size", type=int, default=256, help="Chunk size (default: 256)")
    parser.add_argument(
        "--chunk-overlap", type=int, default=128, help="Chunk overlap (default: 128)"
    )
    parser.add_argument(
        "--enable-code-chunking",
        action="store_true",
        help="Enable AST-aware chunking for code files if astchunk is installed",
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default="./rag_index",
        help="Directory to store the index (default: ./rag_index)",
    )
    parser.add_argument(
        "--index-name",
        type=str,
        default="docs_rag",
        help="Index base name (default: docs_rag)",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=-1,
        help="Maximum number of chunks to index (-1 for all)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Chunks per add batch for progress output (default: 1000)",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to an existing index without prompting (requires non-compact HNSW index)",
    )

    # Embedding parameters
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="facebook/contriever",
        help="Embedding model name (default: facebook/contriever)",
    )
    parser.add_argument(
        "--embedding-mode",
        type=str,
        default="sentence-transformers",
        choices=["sentence-transformers", "openai", "mlx", "ollama"],
        help="Embedding backend mode (default: sentence-transformers)",
    )
    parser.add_argument("--embedding-host", type=str, default=None)
    parser.add_argument("--embedding-api-base", type=str, default=None)
    parser.add_argument("--embedding-api-key", type=str, default=None)

    # Index parameters
    parser.add_argument(
        "--backend-name",
        type=str,
        default="hnsw",
        choices=["hnsw", "diskann"],
        help="Index backend (default: hnsw)",
    )
    parser.add_argument("--graph-degree", type=int, default=32)
    parser.add_argument("--build-complexity", type=int, default=64)
    parser.add_argument("--no-compact", action="store_true")
    parser.add_argument("--no-recompute", action="store_true")

    return parser.parse_args()


def load_documents(args: argparse.Namespace):
    data_path = Path(args.data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")

    reader_kwargs: dict[str, Any] = {"recursive": True, "encoding": "utf-8"}
    if args.file_types:
        reader_kwargs["required_exts"] = args.file_types

    print(f"Loading documents from: {args.data_dir}")
    if args.file_types:
        print(f"Filtering by file types: {args.file_types}")
    documents = SimpleDirectoryReader(args.data_dir, **reader_kwargs).load_data(
        show_progress=True
    )
    print(f"Loaded {len(documents)} documents")
    return documents


def chunk_documents(args: argparse.Namespace, documents):
    print("Chunking documents...")
    chunks = create_text_chunks(
        documents,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        use_ast_chunking=args.enable_code_chunking,
        ast_chunk_size=512,
        ast_chunk_overlap=64,
        code_file_extensions=None,
        ast_fallback_traditional=True,
    )
    if args.max_items > 0 and len(chunks) > args.max_items:
        chunks = chunks[: args.max_items]
    print(f"Prepared {len(chunks)} chunks")
    return chunks


def build_index(args: argparse.Namespace, chunks: list[dict[str, Any] | str]) -> str:
    index_dir = Path(args.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    index_path = str(index_dir / f"{args.index_name}.leann")

    embedding_options: dict[str, Any] = {}
    if args.embedding_mode == "ollama":
        embedding_options["host"] = resolve_ollama_host(args.embedding_host)
    elif args.embedding_mode == "openai":
        embedding_options["base_url"] = resolve_openai_base_url(args.embedding_api_base)
        resolved_key = resolve_openai_api_key(args.embedding_api_key)
        if resolved_key:
            embedding_options["api_key"] = resolved_key

    builder = LeannBuilder(
        backend_name=args.backend_name,
        embedding_model=args.embedding_model,
        embedding_mode=args.embedding_mode,
        embedding_options=embedding_options or None,
        graph_degree=args.graph_degree,
        complexity=args.build_complexity,
        is_compact=not args.no_compact,
        is_recompute=not args.no_recompute,
        num_threads=1,
    )

    total = len(chunks)
    start_time = time.time()
    for start in range(0, total, args.batch_size):
        batch = chunks[start : start + args.batch_size]
        for item in batch:
            if isinstance(item, dict):
                text = item.get("text", "")
                metadata = item.get("metadata")
                builder.add_text(text, metadata)
            else:
                builder.add_text(item)
        done = min(start + args.batch_size, total)
        pct = (done / total * 100) if total else 100.0
        print(f"Added {done}/{total} chunks ({pct:.1f}%)")

    print("Building index (embedding + graph)...")
    builder.build_index(index_path)
    # Create a marker file so index_path exists for loaders that check Path.exists().
    try:
        Path(index_path).touch(exist_ok=True)
    except Exception:
        pass
    elapsed = time.time() - start_time
    print(f"Index saved to: {index_path} (elapsed: {elapsed:.1f}s)")

    register_project_directory(Path.cwd())
    return index_path


def append_index(args: argparse.Namespace, chunks: list[dict[str, Any] | str], index_path: str):
    if args.backend_name != "hnsw":
        raise ValueError("Append is only supported for HNSW backend indices.")
    if not args.no_compact:
        print("Warning: append requires a non-compact HNSW index. Use --no-compact if needed.")

    embedding_options: dict[str, Any] = {}
    if args.embedding_mode == "ollama":
        embedding_options["host"] = resolve_ollama_host(args.embedding_host)
    elif args.embedding_mode == "openai":
        embedding_options["base_url"] = resolve_openai_base_url(args.embedding_api_base)
        resolved_key = resolve_openai_api_key(args.embedding_api_key)
        if resolved_key:
            embedding_options["api_key"] = resolved_key

    builder = LeannBuilder(
        backend_name=args.backend_name,
        embedding_model=args.embedding_model,
        embedding_mode=args.embedding_mode,
        embedding_options=embedding_options or None,
        graph_degree=args.graph_degree,
        complexity=args.build_complexity,
        is_compact=not args.no_compact,
        is_recompute=not args.no_recompute,
        num_threads=1,
    )

    total = len(chunks)
    for start in range(0, total, args.batch_size):
        batch = chunks[start : start + args.batch_size]
        for item in batch:
            if isinstance(item, dict):
                text = item.get("text", "")
                metadata = item.get("metadata")
                builder.add_text(text, metadata)
            else:
                builder.add_text(item)
        done = min(start + args.batch_size, total)
        pct = (done / total * 100) if total else 100.0
        print(f"Prepared {done}/{total} new chunks ({pct:.1f}%)")

    print(f"Appending to existing index: {index_path}")
    builder.update_index(index_path)
    print("Append completed.")


def resolve_index_action(args: argparse.Namespace) -> tuple[str, str]:
    index_dir = Path(args.index_dir)
    index_path = index_dir / f"{args.index_name}.leann"
    if not index_path.exists():
        return "build", str(index_path)
    if args.append:
        return "append", str(index_path)
    if not sys.stdin.isatty():
        raise RuntimeError(
            f"Index already exists at {index_path}. Run with --append or choose a new --index-name."
        )

    while True:
        print(f"Index already exists: {index_path}")
        choice = input("Choose [o]verwrite, [a]ppend, [r]ename, [q]uit: ").strip().lower()
        if choice in {"o", "overwrite"}:
            return "build", str(index_path)
        if choice in {"a", "append"}:
            return "append", str(index_path)
        if choice in {"r", "rename"}:
            new_name = input("New index name (without extension): ").strip()
            if not new_name:
                print("Index name cannot be empty.")
                continue
            args.index_name = new_name
            index_path = Path(args.index_dir) / f"{args.index_name}.leann"
            if not index_path.exists():
                return "build", str(index_path)
            # If the new name also exists, loop again.
            continue
        if choice in {"q", "quit"}:
            raise SystemExit(0)
        print("Invalid choice.")


def main() -> None:
    args = parse_args()
    documents = load_documents(args)
    if not documents:
        print("No documents found.")
        return
    chunks = chunk_documents(args, documents)
    if not chunks:
        print("No chunks to index.")
        return
    action, index_path = resolve_index_action(args)
    if action == "append":
        append_index(args, chunks, index_path)
    else:
        build_index(args, chunks)


if __name__ == "__main__":
    main()
