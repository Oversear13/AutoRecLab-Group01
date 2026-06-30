import os
import json
import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Any

import numpy as np
import pandas as pd

from omnirec import RecSysDataSet, NDCG, HR, Recall
from omnirec.data_loaders.datasets import DataSet
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.split import UserHoldout
from omnirec.runner.algos import LensKit
from omnirec.runner.evaluation import Evaluator
from omnirec.runner.plan import ExperimentPlan
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state


working_dir = os.path.join(os.getcwd(), 'working')
os.makedirs(working_dir, exist_ok=True)


@dataclass
class ExperimentConfig:
    dataset: str
    rating_threshold: float
    core_k: int
    train_frac: float
    val_frac: float
    test_frac: float
    seeds: List[int]
    metrics: List[str]
    algorithms: List[str]


def binarize_ratings_gt_threshold(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    df = df.copy()
    df = df[df['rating'] > threshold].copy()
    df['rating'] = 1
    return df


def get_raw_dataframe(dataset: RecSysDataSet) -> pd.DataFrame:
    for attr in ['data', 'df', 'raw_data']:
        if hasattr(dataset, attr):
            val = getattr(dataset, attr)
            if isinstance(val, pd.DataFrame):
                return val.copy()
    if hasattr(dataset, 'dataframe') and isinstance(dataset.dataframe, pd.DataFrame):
        return dataset.dataframe.copy()
    raise AttributeError('Could not locate raw dataframe on RecSysDataSet; please inspect public dataset accessors.')


def make_dataset_from_dataframe(df: pd.DataFrame) -> RecSysDataSet:
    if hasattr(RecSysDataSet, 'from_dataframe'):
        return RecSysDataSet.from_dataframe(df)
    if hasattr(RecSysDataSet, 'from_df'):
        return RecSysDataSet.from_df(df)
    raise AttributeError('No public dataframe constructor found on RecSysDataSet.')


def build_single_seed_dataset(base_df: pd.DataFrame, seed: int, core_k: int, train_frac: float, val_frac: float) -> RecSysDataSet:
    set_random_state(seed)
    ds = make_dataset_from_dataframe(base_df)
    pipe = Pipe(
        CorePruning(core_k),
        UserHoldout(val_frac, 0.15)
    )
    return pipe.process(ds)


def build_plan() -> ExperimentPlan:
    plan = ExperimentPlan(plan_name='ml100k_seed_sensitivity')
    plan.add_algorithm(LensKit.ImplicitMFScorer, {'feedback': 'implicit'})
    plan.add_algorithm(LensKit.ItemKNNScorer, {'feedback': 'implicit'})
    plan.add_algorithm(LensKit.PopScorer, {'feedback': 'implicit'})
    return plan


def summarize_results(results: Any) -> pd.DataFrame:
    if isinstance(results, pd.DataFrame):
        return results
    if isinstance(results, dict):
        return pd.DataFrame(results)
    return pd.DataFrame([{'result_repr': repr(results)}])


def main():
    config = ExperimentConfig(
        dataset='MovieLens100K',
        rating_threshold=3.0,
        core_k=5,
        train_frac=0.70,
        val_frac=0.15,
        test_frac=0.15,
        seeds=[11, 22, 33, 44, 55],
        metrics=['NDCG@10', 'HR@10', 'Recall@10'],
        algorithms=['ALS(LensKit.ImplicitMFScorer)', 'ItemKNN(LensKit.ItemKNNScorer)', 'Pop(LensKit.PopScorer)']
    )

    dataset = RecSysDataSet.use_dataloader(DataSet.MovieLens100K)
    raw_df = get_raw_dataframe(dataset)
    implicit_df = binarize_ratings_gt_threshold(raw_df, config.rating_threshold)

    base_records = []
    all_seed_results = []
    plan = build_plan()
    evaluator = Evaluator(NDCG([10]), HR([10]), Recall([10]))

    for seed in config.seeds:
        print(f'\n=== Seed {seed} ===')
        split_dataset = build_single_seed_dataset(
            implicit_df,
            seed=seed,
            core_k=config.core_k,
            train_frac=config.train_frac,
            val_frac=config.val_frac,
        )
        set_random_state(seed)
        results = run_omnirec(datasets=split_dataset, plan=plan, evaluator=evaluator)
        results_df = summarize_results(results)
        results_df['seed'] = seed
        all_seed_results.append(results_df)
        base_records.append({
            'seed': seed,
            'dataset': config.dataset,
            'n_interactions_after_threshold': int(len(implicit_df)),
            'split': {'train': config.train_frac, 'validation': config.val_frac, 'test': config.test_frac},
        })
        print(results_df)

    combined = pd.concat(all_seed_results, ignore_index=True) if all_seed_results else pd.DataFrame()
    combined_path = os.path.join(working_dir, 'seed_sensitivity_results.csv')
    combined.to_csv(combined_path, index=False)

    summary = (
        combined.groupby(['algorithm', 'metric'], dropna=False)
        .agg(mean=('value', 'mean'), std=('value', 'std'), min=('value', 'min'), max=('value', 'max'))
        .reset_index()
        if not combined.empty and {'algorithm', 'metric', 'value'}.issubset(combined.columns)
        else pd.DataFrame()
    )
    summary_path = os.path.join(working_dir, 'seed_sensitivity_summary.csv')
    summary.to_csv(summary_path, index=False)

    meta_path = os.path.join(working_dir, 'experiment_config.json')
    with open(meta_path, 'w') as f:
        json.dump({**asdict(config), 'preprocessing': ['ratings > 3 -> implicit', '5-core filtering'], 'seed_records': base_records}, f, indent=2)

    print('\n=== Aggregated Results ===')
    print(summary)
    print(f'\nSaved detailed results to: {combined_path}')
    print(f'Saved summary to: {summary_path}')
    print(f'Saved experiment config to: {meta_path}')


if __name__ == '__main__':
    main()
