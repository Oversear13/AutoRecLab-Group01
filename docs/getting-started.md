# Getting Started

This guide covers the minimum steps needed to run AutoRecLab locally or in Docker.

## Prerequisites

AutoRecLab requires the following:

- Python 3.12 or newer
- One of the following runtime options:
  - uv (recommended)
  - Docker and Docker Compose
  - pip
- Graphviz installed and available on your PATH
- An OpenAI API key

The project also expects documentation embeddings to be available for the documentation index used by the agent.

## Environment setup

### 1. Create an environment file

Create a file named `.env` in the repository root with the following content:

```env
OPENAI_API_KEY=your-openai-api-key
```

If you use Docker, this file is picked up by the Compose configuration.

### 2. Install dependencies

Using uv is the recommended approach:

```bash
uv sync
```

If you prefer pip, you can use:

```bash
pip install -e .
```

## Documentation embeddings

AutoRecLab uses FAISS-based documentation indices to retrieve framework guidance for OmniRec, LensKit, and RecBole. These indices are expected in a directory named `ragEmbeddings`.

Generate them once with:

```bash
uv run python -m cli.embeddings.main generate --all
```

You can also generate only selected sources:

```bash
uv run python -m cli.embeddings.main generate --omnirec --lenskit
```

The generated embeddings are stored in the `ragEmbeddings` directory.

## Running AutoRecLab locally

Start the application with:

```bash
uv run main.py
```

You will then be prompted to enter a research request. Finish the input block with `!start`.

Example:

```text
Enter you request, write "!start" to start:
> Build a reproducible top-N experiment on MovieLens.
> Compare a popularity baseline and a matrix-factorization approach.
> Report Recall@10 and NDCG@10.
> !start
```

## Running AutoRecLab with Docker

A container-based workflow is also available:

```bash
docker compose run --build sandbox
```

The container mounts the project output directory into `./sandbox` on the host system so that generated artifacts remain visible outside the container.

## Initial workspace setup

If you want to create the default workspace folder for output artifacts, you can initialize the workspace explicitly:

```bash
uv run main.py --init
```

## Output folders

By default, AutoRecLab writes its artifacts to `./out`. The most important generated outputs are:

- `summary.md` — final narrative summary of the search outcome
- `save.pkl` — serialized tree state
- `tree_render/` — rendered search-tree visualizations
- `workspace/` — working files used during execution
- `checkpoint/` — per-node execution artifacts

## Common first-run issues

- Missing OpenAI API key: ensure `.env` exists and contains `OPENAI_API_KEY`.
- Missing Graphviz: install `dot` and ensure it is on your PATH.
- Missing documentation embeddings: run the embedding generation step.
- Dependency issues: prefer `uv sync` over plain `pip` when possible.
