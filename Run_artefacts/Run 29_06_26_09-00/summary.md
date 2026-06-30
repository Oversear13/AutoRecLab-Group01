# Experiment Summary

## User Request

The request was to test whether data split random seeds affect recommender-system accuracy on **MovieLens100K** using implicit feedback, after:

1. Converting ratings **greater than 3** to implicit interactions
2. Applying **5-core filtering**
3. Running **5 different random seeds**
4. For each seed, splitting data into **train 0.70 / validation 0.15 / test 0.15**
5. Training **ALS, ItemKNN, and Pop**
6. Evaluating accuracy metrics

## What Was Run

The code performed the requested preprocessing on MovieLens100K:

- `MakeImplicit(3)` converted ratings to implicit feedback using threshold 3
- `CorePruning(5)` applied 5-core filtering

The output confirms preprocessing succeeded:

- Interactions before implicit conversion: **100,000**
- Interactions after implicit conversion: **82,520**
- Interactions after 5-core pruning: **81,697**

The experiment then attempted to run the three algorithms via `ExperimentPlan`:

- `LensKit.ImplicitMFScorer` with `n_factors=50, reg=0.1, iterations=20`
- `LensKit.ItemKNNScorer` with `max_nbrs=50`
- `LensKit.PopScorer`

Metrics requested in the code were:

- `NDCG@10`
- `HR@10`
- `Recall@10`

However, the run failed during the first seed before any algorithm results were produced.

## Key Results

| Item | Result |
|---|---:|
| Dataset | MovieLens100K |
| Implicit threshold | > 3 |
| Interactions before implicit conversion | 100,000 |
| Interactions after implicit conversion | 82,520 |
| Interactions after 5-core pruning | 81,697 |
| Seeds intended | 5 (`11, 22, 33, 44, 55`) |
| Split intended | 0.70 / 0.15 / 0.15 |
| Algorithms intended | ALS, ItemKNN, Pop |
| Metrics intended | NDCG@10, HR@10, Recall@10 |
| Per-seed results | N/A |
| Aggregate results | N/A |

## Limitations

The experiment did **not complete**.

The output shows a failure while creating the runtime environment for `LensKit_env`:

- `CRITICAL Error while creating env 'LensKit_env'`
- The program then crashed with a `SystemExit: 1`

Because the crash occurred at the start of the first seed, there are **no completed splits, trained models, or metric values** available for ALS/ItemKNN/Pop.

Also, although the user requested **ALS**, the code uses `LensKit.ImplicitMFScorer`, which is the algorithm actually configured in the experiment.

## Conclusion

The preprocessing part of the experiment succeeded, and the dataset was reduced to **81,697 implicit interactions** after thresholding and 5-core filtering. But the actual evaluation did **not run to completion**, so there are **no accuracy results** available for comparing the effect of different split random seeds on ALS/ItemKNN/Pop.