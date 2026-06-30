import os
from pathlib import Path

import pandas as pd

from omnirec import RecSysDataSet, NDCG, HR, Recall
from omnirec.data_loaders.datasets import DataSet
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.split import UserHoldout
from omnirec.runner.plan import ExperimentPlan
from omnirec.runner.algos import LensKit
from omnirec.runner.evaluation import Evaluator
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state

SEEDS = [7, 19, 42, 123, 2025]
TOPK = 10


def extract_split_sizes(split_dataset):
    data = split_dataset._data
    train_df = getattr(data, 'train', None)
    valid_df = getattr(data, 'valid', None)
    if valid_df is None:
        valid_df = getattr(data, 'val', None)
    test_df = getattr(data, 'test', None)
    if train_df is None or valid_df is None or test_df is None:
        raise ValueError('Could not access train/valid/test data from OmniRec split dataset.')
    return train_df, valid_df, test_df


def normalize_algorithm_name(algorithm_id):
    if algorithm_id.startswith('LensKit.ImplicitMFScorer'):
        return 'ALS'
    if algorithm_id.startswith('LensKit.ItemKNNScorer'):
        return 'ItemKNN'
    if algorithm_id.startswith('LensKit.PopScorer'):
        return 'Pop'
    return algorithm_id


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

    train_df, valid_df, test_df = extract_split_sizes(split_dataset)
    print(
        f'Split sizes for seed {seed}: '
        f'train={len(train_df)}, valid={len(valid_df)}, test={len(test_df)}'
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
        results_map = evaluator.get_results()
    finally:
        os.chdir(old_cwd)

    if not results_map:
        raise ValueError(f'No evaluation results returned by OmniRec for seed {seed}.')

    dataset_id, long_df = next(iter(results_map.items()))
    if long_df.empty:
        raise ValueError(f'Empty evaluation results for seed {seed} on dataset {dataset_id}.')

    wide_df = (
        long_df.assign(metric=lambda df: df['name'] + '@' + df['k'].astype(int).astype(str))
        .pivot_table(index=['algorithm', 'fold'], columns='metric', values='value', aggfunc='first')
        .reset_index()
    )

    if 'fold' in wide_df.columns:
        wide_df = wide_df.drop(columns=['fold'])

    wide_df['algorithm_raw'] = wide_df['algorithm']
    wide_df['algorithm'] = wide_df['algorithm_raw'].map(normalize_algorithm_name)
    wide_df['seed'] = seed
    wide_df['train_interactions'] = len(train_df)
    wide_df['valid_interactions'] = len(valid_df)
    wide_df['test_interactions'] = len(test_df)

    cols = [
        'seed', 'algorithm', 'algorithm_raw',
        'train_interactions', 'valid_interactions', 'test_interactions',
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

    results_json = Path(seed_dir) / 'omnirec_results.json'
    evaluator.save_results(results_json)

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
    summary_df = (
        results_df.groupby('algorithm', observed=False)[metric_cols]
        .agg(['mean', 'std', 'min', 'max'])
    )
    summary_df.columns = ['_'.join(col) for col in summary_df.columns]
    summary_df = summary_df.reset_index()

    summary_path = os.path.join(working_dir, 'seed_sensitivity_summary.csv')
    summary_df.to_csv(summary_path, index=False)

    print('\n===== Per-seed test results =====')
    print(results_df.to_string(index=False))

    print('\n===== Seed sensitivity summary =====')
    print(summary_df.to_string(index=False))

    print(f'\nSaved per-run results to: {per_run_path}')
    print(f'Saved summary results to: {summary_path}')
