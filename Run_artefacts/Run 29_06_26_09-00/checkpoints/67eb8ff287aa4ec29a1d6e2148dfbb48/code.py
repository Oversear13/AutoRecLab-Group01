import os
import json
import math
import statistics
from collections import defaultdict

import pandas as pd

from omnirec import RecSysDataSet, NDCG, HR, Recall
from omnirec.data_loaders.datasets import DataSet
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.split import RandomHoldout
from omnirec.runner.evaluation import Evaluator
from omnirec.runner.plan import ExperimentPlan
from omnirec.runner.algos import LensKit
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state, get_random_state


def summarize_results(results_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in results_df.columns if c not in {"seed", "algorithm"}]
    agg = results_df.groupby("algorithm")[metric_cols].agg(["mean", "std"]).reset_index()
    return agg


def main():
    working_dir = os.path.join(os.getcwd(), "working")
    os.makedirs(working_dir, exist_ok=True)

    seeds = [11, 22, 33, 44, 55]
    split_proportions = {"train": 0.70, "validation": 0.15, "test": 0.15}

    base_dataset = RecSysDataSet.use_dataloader(DataSet.MovieLens100K)
    pipeline = Pipe(
        MakeImplicit(3),
        CorePruning(5),
    )
    dataset = pipeline.process(base_dataset)

    print("Dataset after preprocessing:")
    print(dataset)
    print(dataset.format_details())

    evaluator = Evaluator(NDCG([10]), HR([10]), Recall([10]))

    plan = ExperimentPlan("MovieLens100K_split_seed_sensitivity")
    plan.add_algorithm(LensKit.ImplicitMFScorer, {"n_factors": 50, "reg": 0.1, "iterations": 20})
    plan.add_algorithm(LensKit.ItemKNNScorer, {"max_nbrs": 50})
    plan.add_algorithm(LensKit.PopScorer, {})

    per_seed_rows = []
    for seed in seeds:
        set_random_state(seed)
        print(f"Running seed={seed}, global_random_state={get_random_state()}")
        split_dataset = RandomHoldout(validation_size=0.15, test_size=0.15).process(dataset)
        seed_results = run_omnirec(datasets=split_dataset, plan=plan, evaluator=evaluator)
        if isinstance(seed_results, pd.DataFrame):
            df = seed_results.copy()
        else:
            df = pd.DataFrame(seed_results)
        df["seed"] = seed
        per_seed_rows.append(df)
        print(df)

    all_results = pd.concat(per_seed_rows, ignore_index=True)
    results_path = os.path.join(working_dir, "split_seed_results.csv")
    all_results.to_csv(results_path, index=False)

    agg = summarize_results(all_results)
    agg_path = os.path.join(working_dir, "split_seed_aggregate.csv")
    agg.to_csv(agg_path, index=False)

    meta = {
        "dataset": "MovieLens100K",
        "implicit_threshold": 3,
        "core": 5,
        "split_proportions": split_proportions,
        "seeds": seeds,
        "algorithms": ["LensKit.ImplicitMFScorer", "LensKit.ItemKNNScorer", "LensKit.PopScorer"],
        "metrics": ["NDCG@10", "HR@10", "Recall@10"],
        "results_path": results_path,
        "aggregate_path": agg_path,
    }
    with open(os.path.join(working_dir, "experiment_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("Per-seed results saved to:", results_path)
    print("Aggregate results saved to:", agg_path)
    print("Metadata saved to:", os.path.join(working_dir, "experiment_metadata.json"))
    print(agg)


if __name__ == "__main__":
    main()
