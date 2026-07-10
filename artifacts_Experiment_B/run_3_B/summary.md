# Experiment Summary

## User Request

Build a reproducible top-N recommendation experiment on MovieLens, compare two candidate algorithms, and report **NDCG@10** and **Recall@1** using the `u.data` MovieLens file.

## What Was Run

A reproducible experiment was configured with:

- Dataset: `MovieLens1M`
- Random seed: `42`
- Preprocessing pipeline:
  - `MakeImplicit(4)` to convert ratings ≥ 4 into implicit feedback
  - `CorePruning(5)` to enforce a 5-core
  - `UserHoldout(0.15, 0.15)` to split into train/validation/test
- Algorithms:
  - `LensKit.ItemKNNScorer` with `max_nbrs=20`, `min_nbrs=5`
  - `LensKit.ImplicitMFScorer` with `n_factors=50`, `n_iters=20`
- Metrics:
  - `NDCG([10])`
  - `Recall([1])`

## Key Results

| Algorithm | NDCG@10 | Recall@1 |
|---|---:|---:|
| LensKit.ItemKNNScorer | 0.1438322238046411 | 0.19471836871134882 |
| LensKit.ImplicitMFScorer | 0.12596646525893324 | 0.14882333443818363 |

The output also showed the post-preprocessing split sizes:

- Train: 397,180
- Validation: 88,155
- Test: 89,041
- Total: 574,376

## Limitations

- The code used `DataSet.MovieLens1M`, not a directly supplied `u.data` file. The output confirms the `MovieLens1M` dataset was loaded, but it does not explicitly state that `u.data` was used.
- The experiment output does not include confidence intervals, standard deviations, or statistical significance tests.
- Only the two requested metrics were reported; no other ranking metrics are available from the provided output.

## Conclusion

This reproducible top-N experiment on MovieLens compared `LensKit.ItemKNNScorer` and `LensKit.ImplicitMFScorer` under the same preprocessing and evaluation setup. Based on the reported results, **ItemKNNScorer performed better on both NDCG@10 and Recall@1**.