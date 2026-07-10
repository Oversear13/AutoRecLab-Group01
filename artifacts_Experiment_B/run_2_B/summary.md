# Experiment Summary

## User Request

Build a reproducible top-N recommendation experiment on MovieLens, compare two candidate algorithms, and report **NDCG@10** and **Recall@1**.

## What Was Run

The experiment code:

- Loaded **MovieLens1M** via `RecSysDataSet.use_dataloader(DataSet.MovieLens1M)`.
- Set a random seed with `set_random_state(42)` for reproducibility.
- Preprocessed the data with:
  - `MakeImplicit(4)` to convert ratings ≥ 4 to implicit feedback.
  - `CorePruning(5)` to enforce 5-core filtering.
  - `UserHoldout(validation_size=0.05, test_size=0.05)` for train/validation/test splitting.
- Compared two algorithms:
  - `LensKit.ItemKNNScorer`
  - `LensKit.ImplicitMFScorer`
- Evaluated with:
  - `NDCG([10])`
  - `Recall([1])`

## Key Results

The run did not complete successfully, so no final metric values were produced.

| Algorithm | NDCG@10 | Recall@1 |
|---|---:|---:|
| ItemKNNScorer | N/A | N/A |
| ImplicitMFScorer | N/A | N/A |

The experiment output shows the runner crashed with:

- `Exception occurred while starting runner: not enough values to unpack (expected 2, got 1)`

Because of this crash, the code never reached the result-printing step.

## Limitations

- No evaluation results were generated, so **NDCG@10** and **Recall@1** cannot be reported from the provided output.
- The crash occurred before `evaluator.get_results()` could return usable metrics.
- The output does not include enough information to identify which exact unpacking operation failed inside the runner.

## Conclusion

This was a reproducible MovieLens1M top-N recommendation experiment setup comparing **ItemKNNScorer** and **ImplicitMFScorer** with **NDCG@10** and **Recall@1** as metrics. However, the experiment **failed during runner startup**, so **no valid metric comparison is available** from the provided run.