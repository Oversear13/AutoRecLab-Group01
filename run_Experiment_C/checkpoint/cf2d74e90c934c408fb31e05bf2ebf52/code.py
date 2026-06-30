import os
from pathlib import Path
import pandas as pd

from omnirec import RecSysDataSet, NDCG, HR, Recall
from omnirec.data_loaders.datasets import DataSet
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.filter import RatingFilter
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.split import RandomHoldout
from omnirec.runner.plan import ExperimentPlan
from omnirec.runner.algos import LensKit
from omnirec.runner.evaluation import Evaluator
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state, get_random_state


SEEDS = [11, 22, 33, 44, 55]
METRICS_TO_TRACK = [("NDCG", 10), ("Recall", 10), ("HR", 10)]


def safe_dataset_id_to_seed_map(dataset_ids, seeds):
    mapping = {}
    for ds_id in dataset_ids:
        matched_seed = None
        for seed in seeds:
            if f"seed{seed}" in ds_id:
                matched_seed = seed
                break
        mapping[ds_id] = matched_seed
    return mapping


def extract_base_algorithm_name(algorithm_identifier):
    if "-" in algorithm_identifier:
        algorithm_identifier = algorithm_identifier.split("-", 1)[0]
    return algorithm_identifier


def preprocess_for_seed(base_dataset, seed, working_dir):
    set_random_state(seed)
    print(f"\n=== Preprocessing for seed {seed} (OmniRec random state={get_random_state()}) ===")

    pipeline = Pipe(
        RatingFilter(lower=4),
        MakeImplicit(1),
        CorePruning(5),
        RandomHoldout(validation_size=0.15, test_size=0.15),
    )
    ds = pipeline.process(base_dataset)

    seed_path = os.path.join(working_dir, f"ml100k_implicit_5core_seed{seed}.rsds")
    ds.save(seed_path)
    if not seed_path.endswith(".rsds"):
        seed_path = seed_path + ".rsds"
    print(f"Saved seed-specific dataset to: {seed_path}")
    return ds, seed_path


def build_plan():
    plan = ExperimentPlan(plan_name="seed_sensitivity_ml100k_implicit")
    plan.add_algorithm(LensKit.ImplicitMFScorer)
    plan.add_algorithm(LensKit.ItemKNNScorer, {"feedback": "implicit"})
    plan.add_algorithm(LensKit.PopScorer)
    return plan


def run_experiment(datasets):
    evaluator = Evaluator(NDCG([10]), HR([10]), Recall([10]))
    plan = build_plan()
    run_omnirec(datasets=datasets, plan=plan, evaluator=evaluator)
    return evaluator


def summarize_results(evaluator, seeds, output_dir):
    results = evaluator.get_results()
    if not results:
        raise RuntimeError("No evaluation results were returned by OmniRec.")

    seed_map = safe_dataset_id_to_seed_map(results.keys(), seeds)
    rows = []
    for dataset_id, df in results.items():
        seed = seed_map.get(dataset_id)
        if seed is None:
            print(f"Warning: could not infer seed from dataset id {dataset_id}; skipping.")
            continue
        temp = df.copy()
        temp["seed"] = seed
        temp["algorithm_base"] = temp["algorithm"].map(extract_base_algorithm_name)
        rows.append(temp)

    if not rows:
        raise RuntimeError("No per-seed rows could be assembled from OmniRec results.")

    all_results = pd.concat(rows, ignore_index=True)
    all_results = all_results[["seed", "algorithm", "algorithm_base", "fold", "name", "k", "value"]]
    all_results = all_results.sort_values(["algorithm_base", "seed", "name", "k"]).reset_index(drop=True)

    per_seed_table = (
        all_results
        .pivot_table(index=["algorithm_base", "seed"], columns=["name", "k"], values="value")
        .reset_index()
    )
    per_seed_table.columns = [
        col if isinstance(col, str) else f"{col[0]}@{col[1]}"
        for col in per_seed_table.columns
    ]
    per_seed_table = per_seed_table.sort_values(["algorithm_base", "seed"]).reset_index(drop=True)

    agg = (
        all_results
        .groupby(["algorithm_base", "name", "k"], as_index=False)["value"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    agg.columns = ["algorithm_base", "name", "k", "mean", "std", "min", "max"]

    summary_table = (
        agg
        .pivot_table(index="algorithm_base", columns=["name", "k"], values=["mean", "std"])
        .sort_index(axis=1)
        .reset_index()
    )
    summary_table.columns = [
        col if isinstance(col, str) else (f"{col[1]}@{col[2]}_{col[0]}" if col[0] else f"{col[1]}@{col[2]}")
        for col in summary_table.columns
    ]

    os.makedirs(output_dir, exist_ok=True)
    all_results_path = os.path.join(output_dir, "all_metric_rows.csv")
    per_seed_path = os.path.join(output_dir, "per_seed_test_results.csv")
    agg_long_path = os.path.join(output_dir, "aggregate_long.csv")
    summary_path = os.path.join(output_dir, "aggregate_wide.csv")

    all_results.to_csv(all_results_path, index=False)
    per_seed_table.to_csv(per_seed_path, index=False)
    agg.to_csv(agg_long_path, index=False)
    summary_table.to_csv(summary_path, index=False)

    print("\n=== Per-seed test results ===")
    print(per_seed_table.to_string(index=False))

    print("\n=== Aggregate seed-sensitivity summary (mean/std across 5 seeds) ===")
    print(summary_table.to_string(index=False))

    print("\nSaved result files:")
    print(f"- {all_results_path}")
    print(f"- {per_seed_path}")
    print(f"- {agg_long_path}")
    print(f"- {summary_path}")

    return all_results, per_seed_table, agg, summary_table


if __name__ == '__main__':
    working_dir = os.path.join(os.getcwd(), 'working')
    os.makedirs(working_dir, exist_ok=True)
    results_dir = os.path.join(working_dir, 'results_seed_sensitivity')
    os.makedirs(results_dir, exist_ok=True)

    print("Loading MovieLens100K with OmniRec...")
    base_dataset = RecSysDataSet.use_dataloader(DataSet.MovieLens100K)
    print(base_dataset)
    print(base_dataset.format_details())

    datasets = []
    saved_paths = []
    for seed in SEEDS:
        ds, pth = preprocess_for_seed(base_dataset, seed, working_dir)
        datasets.append(ds)
        saved_paths.append(pth)

    print("\nPrepared datasets:")
    for seed, pth in zip(SEEDS, saved_paths):
        print(f"  seed={seed}: {pth}")

    print("\nRunning OmniRec experiments for ALS, ItemKNN, and Pop...")
    evaluator = run_experiment(datasets)

    print("\nCollecting and aggregating results...")
    summarize_results(evaluator, SEEDS, results_dir)
