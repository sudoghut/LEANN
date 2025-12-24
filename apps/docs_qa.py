"""
Query a LEANN RAG index built from ./docs (default).
"""

import argparse
import sys
from pathlib import Path
from typing import Any

from leann.api import LeannChat

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
    parser = argparse.ArgumentParser(description="Query a LEANN RAG index.")
    parser.add_argument(
        "--index-path",
        type=str,
        default="./rag_index/docs_rag.leann",
        help="Path to the .leann index (default: ./rag_index/docs_rag.leann)",
    )
    parser.add_argument("--query", type=str, default=None, help="Single query to run")
    parser.add_argument("--top-k", type=int, default=20, help="Top-K results to retrieve")
    parser.add_argument(
        "--search-complexity", type=int, default=32, help="Search complexity (default: 32)"
    )

    # LLM parameters
    parser.add_argument(
        "--llm",
        type=str,
        default="openai",
        choices=["openai", "ollama", "hf", "simulated"],
        help="LLM backend: openai, ollama, hf, or simulated",
    )
    parser.add_argument("--llm-model", type=str, default=None)
    parser.add_argument("--llm-host", type=str, default=None)
    parser.add_argument("--llm-api-base", type=str, default=None)
    parser.add_argument("--llm-api-key", type=str, default=None)
    parser.add_argument(
        "--thinking-budget",
        type=str,
        choices=["low", "medium", "high"],
        default=None,
        help="Thinking budget for reasoning models",
    )

    return parser.parse_args()


def get_llm_config(args: argparse.Namespace) -> dict[str, Any]:
    config: dict[str, Any] = {"type": args.llm}

    if args.llm == "openai":
        config["model"] = args.llm_model or "gpt-4o"
        config["base_url"] = resolve_openai_base_url(args.llm_api_base)
        resolved_key = resolve_openai_api_key(args.llm_api_key)
        if resolved_key:
            config["api_key"] = resolved_key
    elif args.llm == "ollama":
        config["model"] = args.llm_model or "llama3.2:1b"
        config["host"] = resolve_ollama_host(args.llm_host)
    elif args.llm == "hf":
        config["model"] = args.llm_model or "Qwen/Qwen2.5-1.5B-Instruct"
    elif args.llm == "simulated":
        pass

    return config


def run_query(chat: LeannChat, query: str, args: argparse.Namespace) -> str:
    llm_kwargs = {}
    if args.thinking_budget:
        llm_kwargs["thinking_budget"] = args.thinking_budget
    return chat.ask(
        query,
        top_k=args.top_k,
        complexity=args.search_complexity,
        llm_kwargs=llm_kwargs,
    )


def main() -> None:
    args = parse_args()
    index_path = Path(args.index_path)
    if not index_path.exists():
        meta_path = Path(f"{index_path}.meta.json")
        if not meta_path.exists():
            print(f"Index not found: {index_path}")
            sys.exit(1)

    chat = LeannChat(
        str(index_path),
        llm_config=get_llm_config(args),
        complexity=args.search_complexity,
    )

    if args.query:
        response = run_query(chat, args.query, args)
        print(response)
        return

    print("Interactive mode. Type 'exit' to quit.")
    while True:
        query = input("Q> ").strip()
        if not query or query.lower() in {"exit", "quit"}:
            break
        response = run_query(chat, query, args)
        print(f"A> {response}\n")


if __name__ == "__main__":
    main()
