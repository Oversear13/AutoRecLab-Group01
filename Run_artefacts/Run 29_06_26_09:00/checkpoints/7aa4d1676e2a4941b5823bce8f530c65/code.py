import json
import os
import shutil
from pathlib import Path

import pandas as pd

from omnirec import RecSysDataSet, HR, NDCG, Recall
from omnirec.data_loaders.datasets import DataSet
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.split import UserHoldout
from omnirec.runner.algos import LensKit
from omnirec.runner.evaluation import Evaluator
from omnirec.runner.plan import ExperimentPlan
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state, get_random_state


def safe_remove(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def to_dataframe(results):
    if isinstance(results, pd.DataFrame):
        return results.copy()
    return pd.DataFrame(results)


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    metric_cols = [c for c in df.columns if c not in {"seed", "algorithm"}]
    if not metric_cols:
        return df
    grouped = df.groupby("algorithm", dropna=False)[metric_cols].agg(["mean", "std"]).reset_index()
    grouped.columns = ["_".join([str(x) for x in col if x != ""]).strip("_") if isinstance(col, tuple) else col for col in grouped.columns]
    return grouped


def main():
    working_dir = os.path.join(os.getcwd(), "working")
    os.makedirs(working_dir, exist_ok=True)

    # Clean stale runner state that can trigger the Coordinator '_proc' error.
    home = Path.home() / ".omnirec"
    safe_remove(home / "data" / "envs" / "LensKit_env")
    safe_remove(home / "data" / "checkpoints")

    seeds = [11, 22, 33, 44, 55]
    all_rows = []

    base_dataset = RecSysDataSet.use_dataloader(DataSet.MovieLens100K)
    preprocess = Pipe(
        MakeImplicit(3),
        CorePruning(5),
    )
    dataset = preprocess.process(base_dataset)

    print("Dataset after preprocessing:")
    print(dataset)
    print(dataset.format_details())

    evaluator = Evaluator(NDCG([10]), HR([10]), Recall([10]))

    for seed in seeds:
        set_random_state(seed)
        print(f"Running seed={seed}, global_random_state={get_random_state()}")

        # Generate a fresh seed-controlled split for this run.
        split_dataset = UserHoldout(0.15, 0.15).process(dataset)

        plan = ExperimentPlan(plan_name=f"MovieLens100K_seed_{seed}")
        plan.add_algorithm(
            LensKit.ImplicitMFScorer,
            {
                "n_factors": 50,
                "reg": 0.1,
                "iterations": 20,
            },
        )
        plan.add_algorithm(
            LensKit.ItemKNNScorer,
            {
                "max_nbrs": 50,
            },
        )
        plan.add_algorithm(
            LensKit.PopScorer,
            {},
        )

        try:
            results = run_omnirec(datasets=split_dataset, plan=plan, evaluator=evaluator)
            df = to_dataframe(results)
            if not df.empty:
                df["seed"] = seed
                all_rows.append(df)
            print(df)
        except Exception as exc:
            err_df = pd.DataFrame([{"seed": seed, "algorithm": "__error__", "error": repr(exc)}])
            all_rows.append(err_df)
            print(f"Seed {seed} failed: {exc!r}")

    results_path = os.path.join(working_dir, "split_seed_results.csv")
    if all_rows:
        all_results = pd.concat(all_rows, ignore_index=True, sort=False)
    else:
        all_results = pd.DataFrame()
    all_results.to_csv(results_path, index=False)

    completed = all_results[all_results.get("algorithm", pd.Series(dtype=str)) != "__error__"].copy() if not all_results.empty else pd.DataFrame()
    aggregate_path = os.path.join(working_dir, "split_seed_aggregate.csv")
    if not completed.empty:
        agg = summarize_results(completed)
    else:
        agg = pd.DataFrame()
    agg.to_csv(aggregate_path, index=False)

    metadata = {
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
        json.dump(metadata, f, indent=2)

    print("Per-seed results saved to:", results_path)
    print("Aggregate results saved to:", aggregate_path)
    print("Metadata saved to:", metadata_path)
    print(agg)


if __name__ == "__main__":
    main()
