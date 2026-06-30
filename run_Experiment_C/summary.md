# Experiment Summary

## User Request

Quantify how much data split random seeds affect recommender accuracy on implicit-feedback MovieLens100K using three algorithms: ALS, ItemKNN, and Pop.

Requested setup:
- Convert ratings greater than 3 into implicit interactions
- Apply 5-core filtering
- Use 5 different random seeds for data splitting
- Split each run into train 0.70 / validation 0.15 / test 0.15
- Train with standard hyperparameters

## What Was Run

The code ran exactly this pipeline on MovieLens100K:
1. Loaded MovieLens100K
2. Converted ratings to implicit feedback with threshold `> 3`
   - Interactions before conversion: 100,000
   - Interactions after conversion: 82,520
3. Applied 5-core pruning
   - Interactions after pruning: 81,697
4. For each seed in `7, 19, 42, 123, 2025`, applied `UserHoldout(validation_size=0.15, test_size=0.15)`
5. Trained and evaluated:
   - ALS via `LensKit.ImplicitMFScorer`
   - ItemKNN via `LensKit.ItemKNNScorer`
   - Pop via `LensKit.PopScorer`
6. Evaluated test performance with:
   - `NDCG@10`
   - `Recall@10`
   - `HR@10`

Observed split sizes were identical across all 5 seeds:
- Train: 56,419 (`0.690588`)
- Validation: 12,579 (`0.153971`)
- Test: 12,699 (`0.15544`)

## Key Results

The output includes a seed-sensitivity summary across the 5 random seeds. Lower standard deviation/range means less sensitivity to split seed.

| Algorithm | NDCG@10 Mean ± Std | NDCG@10 Range | Recall@10 Mean ± Std | Recall@10 Range | HR@10 Mean ± Std | HR@10 Range |
|---|---:|---:|---:|---:|---:|---:|
| ALS | 0.137413 ± 0.003101 | 0.007615 | 0.192191 ± 0.005515 | 0.014386 | 0.732980 ± 0.012239 | 0.028632 |
| ItemKNN | 0.178910 ± 0.002202 | 0.005431 | 0.226448 ± 0.004266 | 0.011776 | 0.761824 ± 0.016626 | 0.042418 |
| Pop | 0.121243 ± 0.001400 | 0.003485 | 0.146223 ± 0.004352 | 0.010582 | 0.635843 ± 0.015451 | 0.037116 |

Factual takeaways from these results:
- **ItemKNN achieved the best average accuracy** on all three reported metrics:
  - Highest mean `NDCG@10`: `0.178910`
  - Highest mean `Recall@10`: `0.226448`
  - Highest mean `HR@10`: `0.761824`
- **Pop had the lowest average NDCG@10 and Recall@10**, and also lower HR@10 than ALS and ItemKNN.
- **Seed effects were present but modest** across all methods:
  - NDCG@10 standard deviations ranged from `0.001400` to `0.003101`
  - Recall@10 standard deviations ranged from `0.004266` to `0.005515`
  - HR@10 standard deviations ranged from `0.012239` to `0.016626`
- By metric:
  - **ALS showed the largest seed variation** for `NDCG@10` and `Recall@10`
  - **ItemKNN showed the largest seed variation** for `HR@10`
  - **Pop showed the smallest variation** for `NDCG@10`

## Limitations

- The provided output is **truncated**, so the full per-seed metric table is not completely visible.
- Because of that truncation, **exact per-seed results for every algorithm/seed combination cannot be fully reproduced here**.
- However, the final summary statistics and split-size checks are clearly shown, so the conclusions above are grounded in the available output.
- Only one dataset was run in the provided materials: **MovieLens100K**.

## Conclusion

The requested experiment was run on implicit-feedback, 5-core-filtered MovieLens100K with 5 split seeds and standard LensKit-backed models.

Across seeds:
- **ItemKNN was the strongest overall model**
- **ALS was intermediate**
- **Pop was weakest on average accuracy**

Random split seeds did affect accuracy, but the effect was **relatively small in absolute terms** for all three algorithms. Based on the reported standard deviations and ranges, seed sensitivity was noticeable but not large enough to change the overall ranking: **ItemKNN > ALS > Pop** on average.