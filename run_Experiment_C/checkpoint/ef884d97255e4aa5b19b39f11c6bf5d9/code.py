import io
import json
import math
import os
import statistics
import zipfile
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
from omnirec.util.util import set_random_state

SEEDS = [7, 19, 42, 123, 2025]
TOPK = 10


def extract_split_sizes_via_rsds(dataset, save_path):
    dataset.save(str(save_path))
    rsds_path = Path(str(save_path))
    if rsds_path.suffix != '.rsds':
        rsds_path = rsds_path.with_suffix('.rsds')

    if not rsds_path.exists():
        raise FileNotFoundError(f'Expected saved dataset at {rsds_path}')

    with zipfile.ZipFile(rsds_path, 'r') as zf:
        names = zf.namelist()
        data_frames = {}

        for split_name in ('train', 'val', 'valid', 'test'):
            match = None
            for name in names:
                lower = name.lower()
                if split_name in lower and lower.endswith('.csv'):
                    match = name
                    break
            if match is not None:
                with zf.open(match) as fh:
                    data_frames[split_name] = pd.read_csv(fh)

        if 'train' not in data_frames or 'test' not in data_frames:
            manifest = None
            for name in names:
                if name.lower().endswith('.json'):
                    try:
                        with zf.open(name) as fh:
                            manifest = json.load(io.TextIOWrapper(fh, encoding='utf-8'))
                            break
                    except Exception:
                        continue
            raise ValueError(
                'Could not identify split CSV files inside saved .rsds archive. '
                f'Archive entries: {names}. Manifest found: {manifest is not None}'
            )

    valid_key = 'val' if 'val' in data_frames else 'valid'
    if valid_key not in data_frames:
        raise ValueError('Validation split not found in saved .rsds archive.')

    train_df = data_frames['train']
    valid_df = data_frames[valid_key]
    test_df = data_frames['test']
    return train_df, valid_df, test_df


def normalize_algorithm_name(algorithm_id):
    if algorithm_id.startswith('LensKit.ImplicitMFScorer'):
        return 'ALS'
    if algorithm_id.startswith('LensKit.ItemKNNScorer'):
        return 'ItemKNN'
    if algorithm_id.startswith('LensKit.PopScorer'):
        return 'Pop'
    return algorithm_id


def long_results_to_wide(long_df):
    required_cols = {'algorithm', 'name', 'k', 'value'}
    missing = required_cols.difference(long_df.columns)
    if missing:
        raise ValueError(f'Missing expected result columns from OmniRec: {sorted(missing)}')

    work_df = long_df.copy()
    work_df['metric'] = work_df.apply(
        lambda row: f"{row['name']}@{int(row['k'])}" if pd.notna(row['k']) else str(row['name']),
        axis=1,
    )

    wide_df = (
        work_df
        .pivot_table(index='algorithm', columns='metric', values='value', aggfunc='first')
        .reset_index()
    )

    expected_metrics = [f'NDCG@{TOPK}', f'Recall@{TOPK}', f'HR@{TOPK}']
    missing_metrics = [col for col in expected_metrics if col not in wide_df.columns]
    if missing_metrics:
        raise ValueError(
            'Expected metric columns not found after reshaping results. '
            f'Missing: {missing_metrics}. Available: {list(wide_df.columns)}'
        )

    return wide_df


def summarize_seed_sensitivity(results_df, metric_cols):
    rows = []
    for algorithm, group in results_df.groupby('algorithm', observed=False):
        row = {'algorithm': algorithm}
        for metric in metric_cols:
            values = group[metric].tolist()
            row[f'{metric}_mean'] = float(statistics.mean(values))
            row[f'{metric}_std'] = float(statistics.stdev(values)) if len(values) > 1 else 0.0
            row[f'{metric}_min'] = float(min(values))
            row[f'{metric}_max'] = float(max(values))
            row[f'{metric}_range'] = float(max(values) - min(values))
            mean_val = row[f'{metric}_mean']
            row[f'{metric}_cv_pct'] = float((row[f'{metric}_std'] / mean_val) * 100.0) if mean_val != 0 else math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def run_single_seed(seed, working_dir):
    print(f'\n===== Running seed {seed} =====')
    seed_dir = os.path.join(working_dir, f'seed_{seed}')
    os.makedirs(seed_dir, exist_ok=True)

    set_random_state(seed)

    dataset = RecSysDataSet.use_dataloader(DataSet.MovieLens100K)
    print('Loaded dataset:', dataset)

    pipeline = Pipe(
        MakeImplicit(3),
        CorePruning(5),
        UserHoldout(validation_size=0.15, test_size=0.15),
    )
    split_dataset = pipeline.process(dataset)

    split_snapshot = Path(seed_dir) / 'split_dataset_snapshot'
    train_df, valid_df, test_df = extract_split_sizes_via_rsds(split_dataset, split_snapshot)
    total_n = len(train_df) + len(valid_df) + len(test_df)
    print(
        f'Split sizes for seed {seed}: '
        f'train={len(train_df)} ({len(train_df) / total_n:.4f}), '
        f'valid={len(valid_df)} ({len(valid_df) / total_n:.4f}), '
        f'test={len(test_df)} ({len(test_df) / total_n:.4f})'
    )

    plan = ExperimentPlan(plan_name=f'ML100K_seed_{seed}_sensitivity')
    plan.add_algorithm(LensKit.ImplicitMFScorer)
    plan.add_algorithm(LensKit.ItemKNNScorer)
    plan.add_algorithm(LensKit.PopScorer)

    evaluator = Evaluator(
        NDCG([TOPK]),
        HR([TOPK]),
        Recall([TOPK]),
    )

    old_cwd = os.getcwd()
    os.chdir(seed_dir)
    try:
        run_omnirec(datasets=split_dataset, plan=plan, evaluator=evaluator)
        results_json = Path(seed_dir) / 'omnirec_results.json'
        evaluator.save_results(results_json)
        results_map = evaluator.get_results()
    finally:
        os.chdir(old_cwd)

    if not results_map:
        raise ValueError(f'No evaluation results returned by OmniRec for seed {seed}.')

    dataset_id, long_df = next(iter(results_map.items()))
    if long_df.empty:
        raise ValueError(f'Empty evaluation results for seed {seed} on dataset {dataset_id}.')

    wide_df = long_results_to_wide(long_df)
    wide_df['algorithm_raw'] = wide_df['algorithm']
    wide_df['algorithm'] = wide_df['algorithm_raw'].map(normalize_algorithm_name)
    wide_df['seed'] = seed
    wide_df['dataset_id'] = dataset_id
    wide_df['train_interactions'] = len(train_df)
    wide_df['valid_interactions'] = len(valid_df)
    wide_df['test_interactions'] = len(test_df)
    wide_df['train_ratio'] = len(train_df) / total_n
    wide_df['valid_ratio'] = len(valid_df) / total_n
    wide_df['test_ratio'] = len(test_df) / total_n

    cols = [
        'seed', 'dataset_id', 'algorithm', 'algorithm_raw',
        'train_interactions', 'valid_interactions', 'test_interactions',
        'train_ratio', 'valid_ratio', 'test_ratio',
        f'NDCG@{TOPK}', f'Recall@{TOPK}', f'HR@{TOPK}'
    ]
    wide_df = wide_df[cols].sort_values('algorithm').reset_index(drop=True)

    for _, row in wide_df.iterrows():
        print(
            f"Seed {seed} | {row['algorithm']} | "
            f"NDCG@{TOPK}={row[f'NDCG@{TOPK}']:.4f} "
            f"Recall@{TOPK}={row[f'Recall@{TOPK}']:.4f} "
            f"HR@{TOPK}={row[f'HR@{TOPK}']:.4f}"
        )

    return wide_df


if __name__ == '__main__':
    working_dir = os.path.join(os.getcwd(), 'working')
    os.makedirs(working_dir, exist_ok=True)

    all_results = []
    for seed in SEEDS:
        seed_results = run_single_seed(seed, working_dir)
        all_results.append(seed_results)

    results_df = pd.concat(all_results, ignore_index=True)

    preferred_order = ['ALS', 'ItemKNN', 'Pop']
    results_df['algorithm'] = pd.Categorical(
        results_df['algorithm'],
        categories=preferred_order,
        ordered=True,
    )
    results_df = results_df.sort_values(['algorithm', 'seed']).reset_index(drop=True)

    per_run_path = os.path.join(working_dir, 'per_seed_test_results.csv')
    results_df.to_csv(per_run_path, index=False)

    metric_cols = [f'NDCG@{TOPK}', f'Recall@{TOPK}', f'HR@{TOPK}']
    summary_df = summarize_seed_sensitivity(results_df, metric_cols)
    summary_df['algorithm'] = pd.Categorical(
        summary_df['algorithm'],
        categories=preferred_order,
        ordered=True,
    )
    summary_df = summary_df.sort_values('algorithm').reset_index(drop=True)

    summary_path = os.path.join(working_dir, 'seed_sensitivity_summary.csv')
    summary_df.to_csv(summary_path, index=False)

    split_check_df = (
        results_df[
            ['seed', 'train_interactions', 'valid_interactions', 'test_interactions', 'train_ratio', 'valid_ratio', 'test_ratio']
        ]
        .drop_duplicates()
        .sort_values('seed')
        .reset_index(drop=True)
    )
    split_check_path = os.path.join(working_dir, 'split_sizes_by_seed.csv')
    split_check_df.to_csv(split_check_path, index=False)

    print('\n===== Per-seed test results =====')
    print(results_df.to_string(index=False))

    print('\n===== Split sizes by seed =====')
    print(split_check_df.to_string(index=False))

    print('\n===== Seed sensitivity summary =====')
    print(summary_df.to_string(index=False))

    print(f'\nSaved per-run results to: {per_run_path}')
    print(f'Saved split-size checks to: {split_check_path}')
    print(f'Saved summary results to: {summary_path}')
