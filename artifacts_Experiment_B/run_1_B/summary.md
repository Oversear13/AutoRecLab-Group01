# Experiment Summary

## User Request

Build a reproducible top-N recommendation experiment on MovieLens, compare two candidate algorithms, and report NDCG@10 and Recall@1 for MovieLens `u.data`.

## What Was Run

- Dataset: `DataSet.MovieLens100K` from OmniRec, identified in the code as `MovieLens100K u.data`.
- Preprocessing:
  - Converted explicit ratings to implicit feedback with threshold `3`
  - Split with `UserHoldout`
  - Validation size: `0.15`
  - Test size: `0.15`
  - Random seed: `42`
- Algorithms compared:
  - `LensKit.PopScorer`
  - `LensKit.ImplicitMFScorer` with:
    - `n_factors = 64`
    - `n_iters = 20`
    - `reg = 0.1`
- Metrics evaluated:
  - `NDCG@10`
  - `Recall@1`

The output confirms the dataset had 100,000 interactions before implicit conversion and 82,520 after conversion.

## Key Results

| Dataset ID | Algorithm | NDCG@10 | Recall@1 |
|---|---|---:|---:|
| MovieLens100K-78811d22 | LensKit.ImplicitMFScorer-2589c6d7-42 | 0.13198708857921843 | 0.14422057264050903 |
| MovieLens100K-78811d22 | LensKit.PopScorer-f4aa5539-42 | 0.12312375118426273 | 0.17815482502651114 |

## Limitations

- The output reports results for the `MovieLens100K` dataset, which the code labels as `MovieLens100K u.data`, but it does not explicitly print the raw `u.data` filename in the results.
- The algorithm names in the table include run-specific suffixes, so the exact printed identifiers are not the plain class names from the code.
- No additional split-by-user or confidence intervals are provided, so only the single reported mean values can be stated.

## Conclusion

The experiment was run reproducibly with seed `42` on MovieLens 100K using implicit feedback and a user-holdout split.  
Among the two algorithms, `LensKit.ImplicitMFScorer` achieved the higher `NDCG@10`, while `LensKit.PopScorer` achieved the higher `Recall@1`.