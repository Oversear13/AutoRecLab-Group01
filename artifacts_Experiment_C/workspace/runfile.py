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
from omnirec.preprocess.split import RandomHoldout
from omnirec.runner.algos import LensKit
from omnirec.runner.evaluation import Evaluator
from omnirec.runner.plan import ExperimentPlan
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state

SEEDS = [7, 19, 42, 123, 2025]
TOPK = 10
STRICT_GT_3_THRESHOLD = 3.0000001


def rsds_split_tables(rsds_path):
    rsds_path = Path(rsds_path)
    if rsds_path.suffix != '.rsds':
        rsds_path = rsds_path.with_suffix('.rsds')
    if not rsds_path.exists():
        raise FileNotFoundError(f'Missing RSDS file: {rsds_path}')

    tables = {}
    with zipfile.ZipFile(rsds_path, 'r') as zf:
        for name in zf.namelist():
            lower = name.lower()
            if lower.endswith('.csv'):
                key = None
                if 'train' in lower:
                    key = 'train'
                elif 'val' in lower or 'valid' in lower:
                    key = 'val'
                elif 'test' in lower:
                    key = 'test'
                elif lower.endswith('data.csv') or '/data.csv' in lower or lower == 'data.csv':
                    key = 'data'
                if key is not None and key not in tables:
                    with zf.open(name) as fh:
                        tables[key] = pd.read_csv(fh)
    return tables


def save_dataset_and_get_tables(dataset, base_path):
    dataset.save(str(base_path))
    rsds_path = Path(str(base_path))
    if rsds_path.suffix != '.rsds':
        rsds_path = rsds_path.with_suffix('.rsds')
    return rsds_path, rsds_split_tables(rsds_path)


def compute_exact_split_counts(n_interactions):
    train_count = int(round(n_interactions * 0.70))
    val_count = int(round(n_interactions * 0.15))
    test_count = n_interactions - train_count - val_count

    target = {'train': 0.70, 'val': 0.15, 'test': 0.15}
    counts = {'train': train_count, 'val': val_count, 'test': test_count}

    while any(v < 0 for v in counts.values()):
        largest = max(counts, key=lambda part: counts[part])
        counts[largest] -= 1
        counts['test'] += 1

    diff = n_interactions - sum(counts.values())
    if diff != 0:
        residuals = {
            part: target[part] * n_interactions - counts[part]
            for part in counts
        }
        if diff > 0:
            ordered = sorted(residuals, key=lambda p: residuals[p], reverse=True)
            for i in range(diff):
                counts[ordered[i % len(ordered)]] += 1
        else:
            ordered = sorted(residuals, key=lambda p: residuals[p])
            for i in range(-diff):
                counts[ordered[i % len(ordered)]] -= 1

    if sum(counts.values()) != n_interactions:
        raise ValueError('Split counts do not sum to total interactions.')
    if min(counts.values()) < 0:
        raise ValueError(f'Negative split count encountered: {counts}')
    return counts['train'], counts['val'], counts['test']


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


def prepare_exact_random_holdout(seed, seed_dir):
    set_random_state(seed)
    dataset = RecSysDataSet.use_dataloader(DataSet.MovieLens100K)
    print('Loaded dataset:', dataset)

    preprocess = Pipe(
        MakeImplicit(STRICT_GT_3_THRESHOLD),
        CorePruning(5),
    )
    processed = preprocess.process(dataset)

    presplit_base = Path(seed_dir) / 'processed_presplit'
    presplit_rsds, presplit_tables = save_dataset_and_get_tables(processed, presplit_base)

    if 'data' not in presplit_tables:
        reloaded = RecSysDataSet.load(str(presplit_rsds))
        reload_base = Path(seed_dir) / 'processed_presplit_reload'
        _, presplit_tables = save_dataset_and_get_tables(reloaded, reload_base)

    if 'data' not in presplit_tables:
        raise ValueError('Could not recover raw interaction table from saved pre-split dataset.')

    raw_df = presplit_tables['data']
    n_interactions = len(raw_df)
    train_count, val_count, test_count = compute_exact_split_counts(n_interactions)

    print(
        f'Exact target counts for seed {seed}: '
        f'train={train_count}, val={val_count}, test={test_count}, total={n_interactions}'
    )

    split_pipe = Pipe(RandomHoldout(validation_size=val_count, test_size=test_count))
    split_dataset = split_pipe.process(processed)

    split_base = Path(seed_dir) / 'split_dataset'
    _, split_tables = save_dataset_and_get_tables(split_dataset, split_base)

    if not {'train', 'val', 'test'}.issubset(split_tables.keys()):
        raise ValueError(f'Missing split tables in saved split dataset. Found keys: {sorted(split_tables.keys())}')

    train_df = split_tables['train']
    val_df = split_tables['val']
    test_df = split_tables['test']

    actual_counts = (len(train_df), len(val_df), len(test_df))
    expected_counts = (train_count, val_count, test_count)
    if actual_counts != expected_counts:
        raise ValueError(
            'Exact split requirement not met. '
            f'Expected train/val/test={expected_counts}, got {actual_counts}'
        )

    total_n = sum(actual_counts)
    print(
        f'Confirmed split sizes for seed {seed}: '
        f'train={len(train_df)} ({len(train_df) / total_n:.6f}), '
        f'valid={len(val_df)} ({len(val_df) / total_n:.6f}), '
        f'test={len(test_df)} ({len(test_df) / total_n:.6f})'
    )

    return split_dataset, train_df, val_df, test_df


def run_single_seed(seed, working_dir):
    print(f'\n===== Running seed {seed} =====')
    seed_dir = os.path.join(working_dir, f'seed_{seed}')
    os.makedirs(seed_dir, exist_ok=True)

    split_dataset, train_df, valid_df, test_df = prepare_exact_random_holdout(seed, seed_dir)
    total_n = len(train_df) + len(valid_df) + len(test_df)

    plan = ExperimentPlan(plan_name=f'ML100K_seed_{seed}_sensitivity_exact_split')
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