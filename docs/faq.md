# FAQ

## What is AutoRecLab used for?

AutoRecLab is used to turn a natural-language research request into a concrete recommender-systems experiment. It can generate code, execute it, evaluate the outcome, and improve the solution over several iterations.

## Do I need an OpenAI API key?

Yes. The project uses OpenAI-backed LLM calls for planning, code generation, debugging, and documentation-grounded reasoning.

## Why do I need the documentation embeddings?

The documentation embeddings provide a retrieval layer for framework-specific knowledge. They help the agent use the correct APIs and avoid relying purely on model memory.

## What if the embeddings are missing?

If the embeddings are missing, the agent may still run, but it will have less documentation grounding. To fix this, generate the embeddings with:

```bash
uv run python -m cli.embeddings.main generate --all
```

## How do I change the model?

You can override the model either through the CLI or configuration.

Example:

```bash
uv run main.py --model "gpt-4o"
```

You can also set the model in [config.toml](../config.toml) under the `[agent.code]` section.

## What does `out_dir` control?

`out_dir` defines where AutoRecLab writes results such as summaries, checkpoints, tree state, and rendered visualizations. The default is `./out`.

If you want a unique output folder for a single run without changing `config.toml`, use:

```bash
uv run main.py --timestamp-out-dir
```

## How can I inspect the search tree?

After a run, inspect the saved tree state and rendered visualizations in the output folder:

```bash
uv run viz.py -i ./out/save.pkl -o ./out/tree_render
```

## Why does execution sometimes take a long time?

Execution time can increase when:

- the experiment uses large datasets
- the model generates slow or memory-intensive code
- the runtime performs multiple iterations and debugging steps

## Can I run the project without uv?

Yes. The repository also supports Docker and pip-based installation. However, `uv` is the preferred workflow for this project.

## What is the purpose of the type checker?

The optional type checker helps detect obvious Python typing issues before execution. It can improve reliability and provide feedback to the agent when it attempts to fix the code.

## What happens if a generated script fails?

If a generated script fails, AutoRecLab records the error output, marks the candidate as buggy, and attempts to debug or improve it in a subsequent iteration.

## How do I reset or clean previous runs?

You can remove generated output directories such as `out/` and `sandbox/` if you want a fresh run. Make sure you also remove stale embeddings if you want to regenerate them from scratch.

## What if Graphviz is not installed?

The visualization step depends on Graphviz. Install the `dot` executable and ensure it is available in your PATH.
