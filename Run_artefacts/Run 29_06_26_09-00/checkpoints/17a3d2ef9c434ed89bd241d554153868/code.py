import json
import os
import shutil
from pathlib import Path

import pandas as pd

from omnirec import HR, NDCG, Recall, RecSysDataSet
from omnirec.data_loaders.datasets import DataSet
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.split import UserHoldout
from omnirec.runner.algos import LensKit
from omnirec.runner.evaluation import Evaluator
from omnirec.runner.plan import ExperimentPlan
from omnirec.util.run import run_omnirec
from omnirec.util.util import get_random_state, set_random_state


def _to_dataframe(results):
    if isinstance(results, pd.DataFrame):
        return results.copy()
    return pd.DataFrame(results)


def _aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    metric_cols = [c for c in df.columns if c not in {"seed", "algorithm"} and pd.api.types.is_numeric_dtype(df[c])]
    if not metric_cols:
        return pd.DataFrame()
    agg = (
        df.groupby("algorithm", dropna=False)[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    return agg


def _cleanup_omnirec_state():
    # Remove potentially stale env/checkpoint state to avoid coordinator/env reuse issues.
    home = Path.home()
    candidates = [
        home / ".omnirec" / "data" / "envs" / "LensKit_env",
        home / ".omnirec" / "data" / "checkpoints",
        home / ".omnirec" / "data" / "tmp",
    ]
    for path in candidates:
        try:
            if path.exists():
                shutil.rmtree(path)
        except Exception:
            pass


def build_preprocessed_dataset():
    base_dataset = RecSysDataSet.use_dataloader(DataSet.MovieLens100K)
    pipeline = Pipe(
        MakeImplicit(3),
        CorePruning(5),
    )
    return pipeline.process(base_dataset)


def build_plan():
    plan = ExperimentPlan(plan_name="MovieLens100K_split_seed_sensitivity")
    plan.add_algorithm(LensKit.ImplicitMFScorer, {"n_factors": 50, "reg": 0.1, "iterations": 20})
    plan.add_algorithm(LensKit.ItemKNNScorer, {"max_nbrs": 50})
    plan.add_algorithm(LensKit.PopScorer, {})
    return plan


def main():
    working_dir = os.path.join(os.getcwd(), "working")
    os.makedirs(working_dir, exist_ok=True)

    seeds = [11, 22, 33, 44, 55]

    print("Loading and preprocessing MovieLens100K...")
    dataset = build_preprocessed_dataset()
    print(dataset)
    if hasattr(dataset, "format_details"):
        print(dataset.format_details())

    evaluator = Evaluator(NDCG([10]), HR([10]), Recall([10]))

    per_seed_frames = []
    error_frames = []

    for seed in seeds:
        print(f"\n=== Running seed {seed} ===")
        _cleanup_omnirec_state()
        set_random_state(seed)
        print(f"Random state set to: {get_random_state()}")

        # Fresh split object each time; randomness is controlled by the OmniRec global random state.
        split_dataset = UserHoldout(0.15, 0.15).process(dataset)

        plan = build_plan()

        try:
            results = run_omnirec(datasets=split_dataset, plan=plan, evaluator=evaluator)
            df = _to_dataframe(results)
            if not df.empty:
                df = df.copy()
                df["seed"] = seed
                per_seed_frames.append(df)
                print(df)
            else:
                error_frames.append(pd.DataFrame([{ "seed": seed, "algorithm": "__empty__", "error": "No results returned" }]))
                print(f"Seed {seed} returned no results.")
        except Exception as exc:
            print(f"Seed {seed} failed: {exc!r}")
            error_frames.append(pd.DataFrame([{ "seed": seed, "algorithm": "__error__", "error": repr(exc) }]))

    all_results = pd.concat(per_seed_frames + error_frames, ignore_index=True, sort=False) if (per_seed_frames or error_frames) else pd.DataFrame()
    results_path = os.path.join(working_dir, "split_seed_results.csv")
    all_results.to_csv(results_path, index=False)

    completed = all_results[(all_results.get("algorithm", "") != "__error__") & (all_results.get("algorithm", "") != "__empty__")].copy() if not all_results.empty else pd.DataFrame()
    aggregate = _aggregate_results(completed) if not completed.empty else pd.DataFrame()
    aggregate_path = os.path.join(working_dir, "split_seed_aggregate.csv")
    aggregate.to_csv(aggregate_path, index=False)

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

    print("\nPer-seed results saved to:", results_path)
    print("Aggregate results saved to:", aggregate_path)
    print("Metadata saved to:", metadata_path)
    print(aggregate)


if __name__ == "__main__":
    main()
