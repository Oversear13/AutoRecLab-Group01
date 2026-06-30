import os
import shutil
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
TOPK = 10


def extract_base_algorithm_name(algorithm_identifier: str) -> str:
    if algorithm_identifier.startswith("LensKit."):
        core = algorithm_identifier.split(".", 1)[1]
    else:
        core = algorithm_identifier
    core = core.split("-", 1)[0]
    mapping = {
        "ImplicitMFScorer": "ALS",
        "ItemKNNScorer": "ItemKNN",
        "PopScorer": "Pop",
    }
    return mapping.get(core, core)


def clean_old_checkpoints(dataset_names):
    checkpoint_root = Path("checkpoints")
    if not checkpoint_root.exists():
        return
    for name in dataset_names:
        p = checkpoint_root / name
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)


def preprocess_for_seed(base_dataset, seed: int, working_dir: str):
    set_random_state(seed)
    dataset_name = f"MovieLens100K_seed{seed}_implicit_5core"
    print(f"\n=== Preprocessing for seed {seed} (OmniRec random state={get_random_state()}) ===")

    pipeline = Pipe(
        RatingFilter(lower=4),
        MakeImplicit(4),
        CorePruning(5),
        RandomHoldout(validation_size=0.15, test_size=0.15),
    )
    ds = pipeline.process(base_dataset)

    # Give each split dataset a unique public identity so OmniRec keeps separate results/checkpoints.
    ds.meta.name = dataset_name

    save_path = os.path.join(working_dir, dataset_name + ".rsds")
    ds.save(save_path)
    print(f"Saved seed-specific dataset to: {save_path}")
    print(ds)
    return ds, save_path, dataset_name


def build_plan():
    plan = ExperimentPlan(plan_name="seed_sensitivity_ml100k_implicit_unique_datasets")
    plan.add_algorithm(LensKit.ImplicitMFScorer)
    plan.add_algorithm(LensKit.ItemKNNScorer, {"feedback": "implicit"})
    plan.add_algorithm(LensKit.PopScorer)
    return plan


def run_experiment(datasets):
    evaluator = Evaluator(NDCG([TOPK]), HR([TOPK]), Recall([TOPK]))
    plan = build_plan()
    run_omnirec(datasets=datasets, plan=plan, evaluator=evaluator)
    return evaluator


def map_dataset_ids_to_seeds(result_keys, seed_dataset_names):
    mapping = {}
    for dataset_id in result_keys:
        matched_seed = None
        for seed, ds_name in seed_dataset_names.items():
            if dataset_id.startswith(ds_name + "-") or dataset_id == ds_name:
                matched_seed = seed
                break
        mapping[dataset_id] = matched_seed
    return mapping


def summarize_results(evaluator, seed_dataset_names, output_dir):
    results = evaluator.get_results()
    if not results:
        raise RuntimeError("No evaluation results were returned by OmniRec.")

    print("\nAvailable dataset result keys from OmniRec:")
    for key in results.keys():
        print(f"  - {key}")

    seed_map = map_dataset_ids_to_seeds(results.keys(), seed_dataset_names)
    rows = []
    for dataset_id, df in results.items():
        seed = seed_map.get(dataset_id)
        if seed is None:
            print(f"Warning: could not infer seed from dataset id {dataset_id}; skipping.")
            continue
        temp = df.copy()
        temp["seed"] = seed
        temp["dataset_id"] = dataset_id
        temp["algorithm_base"] = temp["algorithm"].map(extract_base_algorithm_name)
        rows.append(temp)

    if not rows:
        raise RuntimeError(
            "No per-seed rows could be assembled from OmniRec results. "
            "Check that each split dataset has a unique dataset.meta.name."
        )

    all_results = pd.concat(rows, ignore_index=True)
    all_results = all_results[["dataset_id", "seed", "algorithm", "algorithm_base", "fold", "name", "k", "value"]]
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
    seed_dataset_names = {}

    for seed in SEEDS:
        ds, pth, ds_name = preprocess_for_seed(base_dataset, seed, working_dir)
        datasets.append(ds)
        saved_paths.append(pth)
        seed_dataset_names[seed] = ds_name

    print("\nPrepared datasets:")
    for seed in SEEDS:
        print(f"  seed={seed}: name={seed_dataset_names[seed]} path={os.path.join(working_dir, seed_dataset_names[seed] + '.rsds')}")

    clean_old_checkpoints(seed_dataset_names.values())

    print("\nRunning OmniRec experiments for ALS, ItemKNN, and Pop...")
    evaluator = run_experiment(datasets)

    print("\nCollecting and aggregating results...")
    summarize_results(evaluator, seed_dataset_names, results_dir)
