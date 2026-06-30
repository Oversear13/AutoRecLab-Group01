import os
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from omnirec import RecSysDataSet
from omnirec import NDCG, HR, Recall
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


def _find_predictions_files(root_dir: str):
    root = Path(root_dir)
    return sorted(root.rglob('predictions.json'))


def _algorithm_name_from_prediction_path(pred_path: Path):
    parent = pred_path.parent.name
    if '-' in parent:
        return parent.split('-')[0]
    return parent


def _load_predictions(pred_path: Path):
    with open(pred_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict):
        # try common payload keys
        for key in ['predictions', 'data', 'rows']:
            if key in data and isinstance(data[key], list):
                data = data[key]
                break
    pred_df = pd.DataFrame(data)
    required_aliases = {
        'user': ['user', 'user_id'],
        'item': ['item', 'item_id'],
        'score': ['score', 'prediction', 'pred'],
        'rank': ['rank']
    }
    rename_map = {}
    for canonical, aliases in required_aliases.items():
        for a in aliases:
            if a in pred_df.columns:
                rename_map[a] = canonical
                break
    pred_df = pred_df.rename(columns=rename_map)
    missing = {'user', 'item'} - set(pred_df.columns)
    if missing:
        raise ValueError(f'Missing required prediction columns {missing} in {pred_path}')
    if 'score' not in pred_df.columns:
        pred_df['score'] = 0.0
    return pred_df


def _get_split_frames(split_dataset):
    data = getattr(split_dataset, '_data')
    if hasattr(data, 'get'):
        train_df = data.get('train')
        val_df = data.get('val')
        test_df = data.get('test')
        return train_df, val_df, test_df
    train_df = getattr(data, 'train', None)
    val_df = getattr(data, 'val', None)
    test_df = getattr(data, 'test', None)
    return train_df, val_df, test_df


def _ranking_metrics_from_predictions(pred_df, test_df, k=10):
    test_truth = test_df.groupby('user')['item'].apply(set).to_dict()
    pred_df = pred_df[pred_df['user'].isin(test_truth.keys())].copy()
    if 'rank' in pred_df.columns:
        pred_df = pred_df.sort_values(['user', 'rank', 'score'], ascending=[True, True, False])
    else:
        pred_df = pred_df.sort_values(['user', 'score'], ascending=[True, False])
    pred_topk = pred_df.groupby('user', as_index=False, group_keys=False).head(k)

    ndcgs = []
    recalls = []
    hrs = []

    for user, group in pred_topk.groupby('user'):
        truth = test_truth.get(user, set())
        if not truth:
            continue
        recs = group['item'].tolist()
        hits = [1 if item in truth else 0 for item in recs]
        dcg = sum(rel / math.log2(idx + 2) for idx, rel in enumerate(hits))
        ideal_len = min(len(truth), k)
        idcg = sum(1.0 / math.log2(idx + 2) for idx in range(ideal_len))
        ndcg = dcg / idcg if idcg > 0 else 0.0
        recall = sum(hits) / len(truth)
        hr = 1.0 if sum(hits) > 0 else 0.0
        ndcgs.append(ndcg)
        recalls.append(recall)
        hrs.append(hr)

    return {
        f'NDCG@{k}': float(np.mean(ndcgs)) if ndcgs else float('nan'),
        f'Recall@{k}': float(np.mean(recalls)) if recalls else float('nan'),
        f'HR@{k}': float(np.mean(hrs)) if hrs else float('nan'),
        'users_evaluated': int(len(ndcgs))
    }


def run_single_seed(seed, working_dir):
    print(f'\n===== Running seed {seed} =====')
    set_random_state(seed)

    dataset = RecSysDataSet.use_dataloader(DataSet.MovieLens100K)
    print('Loaded dataset:', dataset)

    pipeline = Pipe(
        MakeImplicit(3),
        CorePruning(5),
        UserHoldout(0.15, 0.15)
    )
    split_dataset = pipeline.process(dataset)

    train_df, val_df, test_df = _get_split_frames(split_dataset)
    if train_df is None or val_df is None or test_df is None:
        raise ValueError('Could not access train/val/test splits from OmniRec SplitData.')

    print(f'Split sizes for seed {seed}: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}')

    seed_dir = os.path.join(working_dir, f'seed_{seed}')
    if os.path.exists(seed_dir):
        shutil.rmtree(seed_dir)
    os.makedirs(seed_dir, exist_ok=True)

    old_cwd = os.getcwd()
    os.chdir(seed_dir)
    try:
        plan = ExperimentPlan(plan_name=f'ML100K_seed_{seed}_sensitivity')
        plan.add_algorithm(LensKit.ImplicitMFScorer, {})
        plan.add_algorithm(LensKit.ItemKNNScorer, {'feedback': 'implicit'})
        plan.add_algorithm(LensKit.PopScorer, {})

        evaluator = Evaluator(
            NDCG([TOPK]),
            Recall([TOPK]),
            HR([TOPK])
        )

        run_omnirec(datasets=split_dataset, plan=plan, evaluator=evaluator)
    finally:
        os.chdir(old_cwd)

    pred_files = _find_predictions_files(seed_dir)
    pred_files = [p for p in pred_files if 'fold_' not in str(p)]
    if not pred_files:
        raise FileNotFoundError(f'No OmniRec predictions.json files found under {seed_dir}')

    rows = []
    for pred_path in pred_files:
        algo = _algorithm_name_from_prediction_path(pred_path)
        pred_df = _load_predictions(pred_path)
        metrics = _ranking_metrics_from_predictions(pred_df, test_df, k=TOPK)
        row = {
            'seed': seed,
            'algorithm': algo,
            'train_interactions': len(train_df),
            'val_interactions': len(val_df),
            'test_interactions': len(test_df),
        }
        row.update(metrics)
        rows.append(row)
        print(f"Seed {seed} | {algo} | "
              f"NDCG@{TOPK}={row[f'NDCG@{TOPK}']:.4f} "
              f"Recall@{TOPK}={row[f'Recall@{TOPK}']:.4f} "
              f"HR@{TOPK}={row[f'HR@{TOPK}']:.4f}")

    return pd.DataFrame(rows)


if __name__ == '__main__':
    working_dir = os.path.join(os.getcwd(), 'working')
    os.makedirs(working_dir, exist_ok=True)

    all_results = []
    for seed in SEEDS:
        seed_results = run_single_seed(seed, working_dir)
        all_results.append(seed_results)

    results_df = pd.concat(all_results, ignore_index=True)

    preferred_order = ['ImplicitMFScorer', 'ItemKNNScorer', 'PopScorer']
    results_df['algorithm'] = pd.Categorical(results_df['algorithm'], categories=preferred_order, ordered=True)
    results_df = results_df.sort_values(['algorithm', 'seed']).reset_index(drop=True)

    per_run_path = os.path.join(working_dir, 'per_seed_test_results.csv')
    results_df.to_csv(per_run_path, index=False)

    metric_cols = [f'NDCG@{TOPK}', f'Recall@{TOPK}', f'HR@{TOPK}']
    summary_df = results_df.groupby('algorithm')[metric_cols].agg(['mean', 'std', 'min', 'max'])
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
