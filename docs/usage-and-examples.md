# Usage and Examples

AutoRecLab can be used either interactively or through non-interactive CLI arguments.

## Starting the application

### Interactive mode

```bash
uv run main.py
```

Then enter a prompt and finish with `!start`.

### Non-interactive mode with a prompt

```bash
uv run main.py --prompt "Analyze the signal and generate a report"
```

### Load a prompt from a file

```bash
uv run main.py --prompt-file ./my-prompt.txt
```

### Initialize the workspace

```bash
uv run main.py --init
```

### List available datasets

```bash
uv run main.py --list-datasets
```

### List available models

```bash
uv run main.py --list-models
```

### Override the model

```bash
uv run main.py --model "gpt-4o"
```

## Example prompts

Here are several prompts that work well with AutoRecLab.

### Example 1: Baseline experiment

```text
Build a reproducible top-N recommendation experiment on MovieLens.
Compare a popularity baseline and a matrix-factorization approach.
Report Recall@10 and NDCG@10.
```

### Example 2: Dataset comparison

```text
Compare two recommendation models on the Amazon Books dataset.
Use a simple but reproducible evaluation protocol.
Report metrics with confidence or variance estimates where practical.
```

### Example 3: Research-style experiment

```text
Investigate whether a simple collaborative filtering baseline outperforms a popularity baseline on a sparse implicit-feedback dataset.
Use only a small and reproducible setup suitable for quick experimentation.
```

### Example 4: Multi-step study

```text
Design a small experiment that compares a content-based baseline against a collaborative approach.
Keep the implementation simple, use the provided dataset, and summarize the results clearly.
```

## Example command sequence

```bash
uv sync
uv run python -m cli.embeddings.main generate --all
uv run main.py --prompt "Build a reproducible recommendation experiment on a small dataset and report ranking metrics."
```

## Output interpretation

After a run completes, AutoRecLab typically writes:

- a final summary in `out/summary.md`
- a saved search tree in `out/save.pkl`
- rendered tree images in `out/tree_render/`
- generated code artifacts under `out/checkpoint/`

## Visualization

If you want to render the saved tree structure after a run, use:

```bash
uv run viz.py -i ./out/save.pkl -o ./out/tree_render
```

This creates graph files that make it easier to inspect how the system explored candidate solutions.

## Suggested workflow

1. Start with a clear and narrow prompt.
2. Ensure the required documentation embeddings exist.
3. Run the experiment and inspect the generated summary.
4. Review the tree visualization if the run involved several iterations.
5. Refine the prompt if you want a more detailed or more structured experiment.
