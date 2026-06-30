import os
import json
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


def _flatten_results(results_obj):
    if isinstance(results_obj, dict):
        frames = []
        for key, value in results_obj.items():
            if isinstance(value, pd.DataFrame):
                df = value.copy()
            else:
                df = pd.DataFrame(value)
            df["dataset_id"] = key
            frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if isinstance(results_obj, pd.DataFrame):
        return results_obj.copy()
    return pd.DataFrame(results_obj)


def aggregate_across_seeds(df):
    metric_cols = [c for c in df.columns if c not in {"seed", "dataset_id"}]
    group_cols = [c for c in ["algorithm", "name", "k"] if c in df.columns]
    if not group_cols:
        return pd.DataFrame()
    return df.groupby(group_cols, as_index=False)[metric_cols].mean(numeric_only=True)


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

    evaluator = Evaluator(
        NDCG([10]),
        HR([10]),
        Recall([10]),
    )

    per_seed_frames = []

    for seed in seeds:
        set_random_state(seed)
        print(f"Running seed={seed}, global_random_state={get_random_state()}")

        split_dataset = RandomHoldout(validation_size=0.15, test_size=0.15).process(dataset)

        plan = ExperimentPlan(f"MovieLens100K_seed_{seed}")
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

        run_omnirec(datasets=split_dataset, plan=plan, evaluator=evaluator)
        seed_results = evaluator.get_results()
        seed_df = _flatten_results(seed_results)
        seed_df["seed"] = seed
        per_seed_frames.append(seed_df)
        print(seed_df)

    all_results = pd.concat(per_seed_frames, ignore_index=True) if per_seed_frames else pd.DataFrame()
    results_path = os.path.join(working_dir, "split_seed_results.csv")
    all_results.to_csv(results_path, index=False)

    agg = aggregate_across_seeds(all_results)
    agg_path = os.path.join(working_dir, "split_seed_aggregate.csv")
    agg.to_csv(agg_path, index=False)

    meta = {
        "dataset": "MovieLens100K",
        "implicit_threshold": 3,
        "core": 5,
        "split_proportions": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "seeds": seeds,
        "algorithms": ["LensKit.ImplicitMFScorer", "LensKit.ItemKNNScorer", "LensKit.PopScorer"],
        "metrics": ["NDCG@10", "HR@10", "Recall@10"],
        "results_path": results_path,
        "aggregate_path": agg_path,
    }
    meta_path = os.path.join(working_dir, "experiment_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print("Per-seed results saved to:", results_path)
    print("Aggregate results saved to:", agg_path)
    print("Metadata saved to:", meta_path)
    print(agg)


if __name__ == "__main__":
    main()
