import os
import json
import matplotlib.pyplot as plt
import pandas as pd

from omnirec import RecSysDataSet, NDCG
from omnirec.data_loaders.datasets import DataSet
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.split import UserHoldout
from omnirec.runner.plan import ExperimentPlan
from omnirec.runner.evaluation import Evaluator
from omnirec.runner.algos import LensKit
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state


def _extract_metric(results, metric_name):
    if results is None:
        return None
    try:
        if hasattr(results, 'columns') and metric_name in results.columns:
            return float(results.iloc[0][metric_name])
        if isinstance(results, dict) and metric_name in results:
            return float(results[metric_name])
    except Exception:
        return None
    return None


def _split_stats(dataset):
    if not hasattr(dataset, '_data'):
        return {}
    stats = {}
    for split_name in ('train', 'val', 'test'):
        try:
            split_df = dataset._data.get(split_name)
            if split_df is not None:
                stats[f'{split_name}_interactions'] = int(len(split_df))
        except Exception:
            pass
    return stats


def main():
    working_dir = os.path.join(os.getcwd(), 'working')
    os.makedirs(working_dir, exist_ok=True)

    set_random_state(42)

    dataset = RecSysDataSet.use_dataloader(DataSet.MovieLens1M)
    print('Loaded dataset:', dataset.meta.name)
    print('Raw interactions:', dataset.num_interactions())

    pipeline = Pipe(
        MakeImplicit(4),
        UserHoldout(0.2, 0.2),
    )
    dataset = pipeline.process(dataset)

    stats = {'dataset': dataset.meta.name, **_split_stats(dataset)}
    stats['total_interactions'] = sum(v for k, v in stats.items() if k.endswith('_interactions'))
    print('Post-preprocessing stats:', stats)

    plan = ExperimentPlan(plan_name='movielens1m_prototype')
    plan.add_algorithm(
        LensKit.PopScorer,
        {}
    )

    evaluator = Evaluator(NDCG([10]))
    results = run_omnirec(datasets=dataset, plan=plan, evaluator=evaluator)
    print(results)

    results_path = os.path.join(working_dir, 'prototype_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        try:
            json.dump(str(results), f, indent=2)
        except Exception:
            f.write(str(results))

    ndcg10 = _extract_metric(results, 'NDCG@10')
    print({'NDCG@10': ndcg10})

    plot_path = os.path.join(working_dir, 'prototype_metrics.png')
    plt.figure(figsize=(5, 4))
    plt.bar(['NDCG@10'], [ndcg10 if ndcg10 is not None else 0.0])
    plt.ylabel('Score')
    plt.title('Prototype Metric on MovieLens1M')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    print(f'Saved plot to {plot_path}')


if __name__ == '__main__':
    main()
