import os
import json
from pathlib import Path

import matplotlib.pyplot as plt

from omnirec import RecSysDataSet, NDCG, Recall
from omnirec.data_loaders.datasets import DataSet
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.split import UserHoldout
from omnirec.runner.plan import ExperimentPlan
from omnirec.runner.algos import LensKit
from omnirec.runner.evaluation import Evaluator
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state


def extract_metric_value(results, metric_name):
    try:
        if hasattr(results, 'columns') and metric_name in results.columns:
            row = results.iloc[0]
            return float(row[metric_name])
    except Exception:
        pass
    try:
        if isinstance(results, dict) and metric_name in results:
            return float(results[metric_name])
    except Exception:
        pass
    return 0.0


def main():
    working_dir = os.path.join(os.getcwd(), 'working')
    os.makedirs(working_dir, exist_ok=True)

    set_random_state(42)

    dataset = RecSysDataSet.use_dataloader(DataSet.MovieLens100K)
    raw_stats = {
        'num_interactions': int(dataset.num_interactions()),
        'min_rating': float(dataset.min_rating()),
        'max_rating': float(dataset.max_rating()),
    }

    pipeline = Pipe(
        UserHoldout(validation_size=0.15, test_size=0.15),
    )
    dataset = pipeline.process(dataset)

    post_stats = {
        'num_interactions': int(dataset.num_interactions()),
        'min_rating': float(dataset.min_rating()),
        'max_rating': float(dataset.max_rating()),
    }

    plan = ExperimentPlan(plan_name='movielens100k_prototype')
    plan.add_algorithm(LensKit.PopScorer, {})

    evaluator = Evaluator(NDCG([10]), Recall([1]))

    results = run_omnirec(datasets=dataset, plan=plan, evaluator=evaluator)
    print(results)

    results_path = os.path.join(working_dir, 'prototype_results.json')
    payload = {
        'raw_stats': raw_stats,
        'post_stats': post_stats,
        'results_repr': str(results),
    }
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)

    ndcg10 = extract_metric_value(results, 'NDCG@10')
    recall1 = extract_metric_value(results, 'Recall@1')

    plot_path = os.path.join(working_dir, 'prototype_metrics.png')
    plt.figure(figsize=(6, 4))
    plt.bar(['NDCG@10', 'Recall@1'], [ndcg10, recall1])
    plt.ylabel('Score')
    plt.title('MovieLens100K Prototype Metrics')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    print(f'Saved plot to {plot_path}')
    print('Raw dataset stats:', raw_stats)
    print('Post-preprocessing stats:', post_stats)
    print({'NDCG@10': ndcg10, 'Recall@1': recall1})


if __name__ == '__main__':
    main()
