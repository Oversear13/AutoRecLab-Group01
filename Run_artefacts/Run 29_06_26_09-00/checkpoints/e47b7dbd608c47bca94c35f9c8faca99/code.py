import os
import json
import pandas as pd

from omnirec import RecSysDataSet, NDCG, HR, Recall
from omnirec.data_loaders.datasets import DataSet
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.split import UserHoldout
from omnirec.runner.evaluation import Evaluator
from omnirec.runner.plan import ExperimentPlan
from omnirec.runner.algos import LensKit
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state, get_random_state


def normalize_results(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "seed" not in df.columns:
        raise ValueError("Expected a 'seed' column in results")
    metric_cols = [c for c in df.columns if c not in {"seed", "algorithm"}]
    agg = df.groupby("algorithm", dropna=False)[metric_cols].agg(["mean", "std"]).reset_index()
    return agg


def extract_results(results):
    if isinstance(results, pd.DataFrame):
        return results.copy()
    return pd.DataFrame(results)


def main():
    working_dir = os.path.join(os.getcwd(), "working")
    os.makedirs(working_dir, exist_ok=True)

    seeds = [11, 22, 33, 44, 55]

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

    per_seed_rows = []
    all_seed_summaries = []

    for seed in seeds:
        set_random_state(seed)
        print(f"Running seed={seed}, global_random_state={get_random_state()}")

        split_dataset = UserHoldout(0.15, 0.15).process(dataset)

        plan = ExperimentPlan(plan_name=f"MovieLens100K_seed_{seed}")
        plan.add_algorithm(LensKit.ImplicitMFScorer, {"n_factors": 50, "reg": 0.1, "iterations": 20})
        plan.add_algorithm(LensKit.ItemKNNScorer, {"max_nbrs": 50})
        plan.add_algorithm(LensKit.PopScorer, {})

        try:
            seed_results = run_omnirec(datasets=split_dataset, plan=plan, evaluator=evaluator)
            df = extract_results(seed_results)
            df["seed"] = seed
            per_seed_rows.append(df)
            all_seed_summaries.append(df)
            print(df)
        except Exception as exc:
            # Keep the experiment running so we still get the remaining seeds.
            err_df = pd.DataFrame(
                [{"seed": seed, "algorithm": "__error__", "error": repr(exc)}]
            )
            per_seed_rows.append(err_df)
            print(f"Seed {seed} failed: {exc!r}")

    all_results = pd.concat(per_seed_rows, ignore_index=True, sort=False)
    results_path = os.path.join(working_dir, "split_seed_results.csv")
    all_results.to_csv(results_path, index=False)

    completed_rows = all_results[all_results.get("algorithm", "") != "__error__"].copy()
    aggregate_path = os.path.join(working_dir, "split_seed_aggregate.csv")
    if not completed_rows.empty:
        agg = normalize_results(completed_rows)
        agg.to_csv(aggregate_path, index=False)
    else:
        agg = pd.DataFrame()
        agg.to_csv(aggregate_path, index=False)

    meta = {
        "dataset": "MovieLens100K",
        "implicit_threshold": 3,
        "core": 5,
        "seeds": seeds,
        "split_proportions": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "algorithms": ["LensKit.ImplicitMFScorer", "LensKit.ItemKNNScorer", "LensKit.PopScorer"],
        "metrics": ["NDCG@10", "HR@10", "Recall@10"],
        "results_path": results_path,
        "aggregate_path": aggregate_path,
    }
    metadata_path = os.path.join(working_dir, "experiment_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(meta, f, indent=2)

    print("Per-seed results saved to:", results_path)
    print("Aggregate results saved to:", aggregate_path)
    print("Metadata saved to:", metadata_path)
    print(agg)


if __name__ == "__main__":
    main()
