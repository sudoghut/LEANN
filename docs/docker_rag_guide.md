# Docker Compose RAG Guide

This guide documents the Docker Compose setup for running the RAG scripts with GPU support.

## Files Added

- `Dockerfile.rag`
- `docker-compose.yml`

## What It Does

- Builds a Python 3.11 image with CUDA-enabled PyTorch and RAG dependencies.
- Runs the existing scripts against your local workspace via a bind mount.
- Enables GPU access using Docker deploy device reservations for NVIDIA GPUs.
- The image has no default command; always run with an explicit command.

## Build and Run

```bash
# Build the image
docker compose build

# Build the index (reads ./knowledge-base by default, writes ./rag_index/docs_rag.leann)
docker compose run --rm leann-rag python apps/docs_embed.py

# Filter file types / chunk settings
docker compose run --rm leann-rag python apps/docs_embed.py --file-types .pdf .md --chunk-size 400 --chunk-overlap 120

# Append without prompt (requires non-compact HNSW index)
docker compose run --rm leann-rag python apps/docs_embed.py --append --no-compact

# Ask a single question
docker compose run --rm leann-rag python apps/docs_qa.py --query "你的问题"

# Ask using a specific index
docker compose run --rm leann-rag python apps/docs_qa.py --index-path ./rag_index/your_index.leann --query "你的问题"

# Interactive Q&A
docker compose run --rm leann-rag python apps/docs_qa.py
```

## Local (uv run) Reference

```bash
# Build the index (reads ./knowledge-base by default, writes ./rag_index/docs_rag.leann)
uv run python apps/docs_embed.py

# Filter file types / chunk settings
uv run python apps/docs_embed.py --file-types .pdf .md --chunk-size 400 --chunk-overlap 120

# Append without prompt (requires non-compact HNSW index)
uv run python apps/docs_embed.py --append --no-compact

# Ask a single question
uv run python apps/docs_qa.py --query "你的问题"

# Ask using a specific index
uv run python apps/docs_qa.py --index-path ./rag_index/your_index.leann --query "你的问题"

# Interactive Q&A
uv run python apps/docs_qa.py
```

## Requirements

- Windows + WSL2
- Docker Desktop with WSL2 engine enabled
- NVIDIA GPU drivers installed on Windows

## Notes

- The project directory is mounted into `/workspace` inside the container.
- The default index output is `./rag_index/docs_rag.leann` in the repo.
- If your Docker Compose ignores `deploy` GPU reservations, run with `docker compose run --rm --gpus all ...`.
- If you need a different Python package mirror, update `Dockerfile.rag` to add `-i <mirror>` to the `pip install` commands.
- If the index already exists, the embed script prompts to overwrite, append, or rename. Use `--append` to skip the prompt.
- Some builds may not create a physical `.leann` file; `docs_qa.py` accepts the index if the matching `.leann.meta.json` exists.

## Ollama in Docker

When running `docs_qa.py` inside Docker, `localhost` points to the container, not your host.
Use `host.docker.internal` and ensure Ollama is running on the host.

```bash
# Start Ollama on the host
ollama serve

# Run QA with Ollama from the container
docker compose run --rm leann-rag \
  python apps/docs_qa.py --llm ollama --llm-model llama3.2:1b \
  --llm-host http://host.docker.internal:11434
```

One-liner:

```bash
docker compose run --rm leann-rag python apps/docs_qa.py --llm ollama --llm-model llama3.2:1b --llm-host http://host.docker.internal:11434
```

## OpenAI in Docker

Provide an API key via environment variable or `--llm-api-key`.

```bash
# Using environment variable
docker compose run --rm -e OPENAI_API_KEY=your_key leann-rag \
  python apps/docs_qa.py --llm openai --query "你的问题"

# Using explicit parameter
docker compose run --rm leann-rag \
  python apps/docs_qa.py --llm openai --llm-api-key your_key --query "你的问题"
```
