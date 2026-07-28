# AutoRecLab Documentation

AutoRecLab is an autonomous research agent for recommender-systems experimentation. It translates a natural-language research task into executable Python, evaluates the result, and iteratively improves the solution through a tree-search loop.

This documentation provides a practical and technical overview of the project, including setup, architecture, usage patterns, examples, and frequently asked questions.

## What AutoRecLab does

At a high level, AutoRecLab performs the following workflow:

1. Receives a research request from the user.
2. Converts the request into concrete experiment requirements.
3. Generates candidate implementations with the help of an LLM.
4. Executes the code in a controlled workspace.
5. Scores the results, analyzes failures, and iteratively improves the solution.
6. Produces a final summary and stores artifacts for inspection.

## Key capabilities

- Natural-language to executable experiment generation
- Iterative improvement through a search tree
- Execution and debugging loops
- Optional type checking before execution
- Documentation-aware retrieval through vector stores and MCP tools
- Dataset discovery and experiment artifact generation
- Tree visualization for inspection of intermediate results

## Documentation map

- [Getting Started](getting-started.md) — prerequisites, installation, first run, Docker setup
- [Architecture](architecture.md) — main components and internal execution flow
- [Usage and Examples](usage-and-examples.md) — CLI usage, prompts, and sample workflows
- [FAQ](faq.md) — common questions and troubleshooting guidance

## Repository layout

The main project areas are:

- [main.py](../main.py) — command-line entry point
- [config.py](../config.py) — configuration loading and environment overrides
- [config.toml](../config.toml) — default runtime configuration
- [treesearch/](../treesearch) — search logic, agents, execution, and scoring
- [cli/embeddings/](../cli/embeddings) — documentation embedding generation
- [utils/](../utils) — logging, path utilities, and statistics tracking
- [viz.py](../viz.py) — visualization of the search tree

## Recommended first read

If you are new to the project, start with [Getting Started](getting-started.md), then read [Architecture](architecture.md) and [Usage and Examples](usage-and-examples.md).
