import os
import json
import matplotlib.pyplot as plt

from omnirec import RecSysDataSet
from omnirec.data_loaders.datasets import DataSet
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.split import UserHoldout
from omnirec.runner.plan import ExperimentPlan
from omnirec.runner.evaluation import Evaluator
from omnirec.runner.algos import LensKit
from omnirec.metrics.ranking import NDCG, Recall
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state


def _extract_results_dataframe(results):
    if results is None:
        return None
    if isinstance(results, dict):
        # OmniRec documented return type: dict of dataset_id -> DataFrame
        for df in results.values():
            return df
    return results


def _get_metric_value(results_df, algorithm_name, metric_name, k):
    try:
        if results_df is None:
            return None
        df = results_df
        if not hasattr(df, 'columns'):
            return None
        subset = df[
            (df['algorithm'].astype(str).str.contains(algorithm_name, regex=False))
            & (df['name'] == metric_name)
            & (df['k'] == k)
        ]
        if subset.empty:
            return None
        return float(subset['value'].mean())
    except Exception:
        return None


def main():
    working_dir = os.path.join(os.getcwd(), 'working')
    os.makedirs(working_dir, exist_ok=True)

    set_random_state(42)

    dataset = RecSysDataSet.use_dataloader(DataSet.MovieLens1M)
    print('Loaded dataset:', dataset.meta.name)
    print('Raw interactions:', dataset.num_interactions())

    pipeline = Pipe(
        MakeImplicit(4),
        CorePruning(5),
        UserHoldout(0.15, 0.15),
    )
    dataset = pipeline.process(dataset)

    train_df = dataset._data.get('train')
    val_df = dataset._data.get('val')
    test_df = dataset._data.get('test')
    stats = {
        'dataset': dataset.meta.name,
        'train_interactions': int(len(train_df)),
        'val_interactions': int(len(val_df)),
        'test_interactions': int(len(test_df)),
        'total_interactions': int(len(train_df) + len(val_df) + len(test_df)),
    }
    print('Post-preprocessing stats:', stats)

    plan = ExperimentPlan(plan_name='movielens1m_prototype')
    plan.add_algorithm(
        LensKit.ItemKNNScorer,
        {'max_nbrs': 20, 'min_nbrs': 5}
    )
    plan.add_algorithm(
        LensKit.ImplicitMFScorer,
        {'n_factors': 50, 'n_iters': 20}
    )

    experiment_metadata = {
        'dataset': dataset.meta.name,
        'split_protocol': 'MakeImplicit(4) -> CorePruning(5) -> UserHoldout(0.15, 0.15)',
        'algorithms': ['LensKit.ItemKNNScorer', 'LensKit.ImplicitMFScorer'],
        'metrics': ['NDCG@10', 'Recall@1'],
    }
    print('Experiment metadata:', experiment_metadata)

    evaluator = Evaluator(NDCG([10]), Recall([1]))
    results = run_omnirec(datasets=dataset, plan=plan, evaluator=evaluator)
    print(results)

    results_df = _extract_results_dataframe(evaluator.get_results())
    if results_df is None:
        structured_results = {}
    else:
        structured_results = {
            'dataset': dataset.meta.name,
            'results': results_df.to_dict(orient='records'),
        }

    results_path = os.path.join(working_dir, 'prototype_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(structured_results, f, indent=2)

    algorithm_names = ['LensKit.ItemKNNScorer', 'LensKit.ImplicitMFScorer']
    summary = {}
    for algo in algorithm_names:
        summary[algo] = {
            'NDCG@10': _get_metric_value(results_df, algo, 'NDCG', 10),
            'Recall@1': _get_metric_value(results_df, algo, 'Recall', 1),
        }
    print(summary)

    plot_path = os.path.join(working_dir, 'prototype_metrics.png')
    labels = ['NDCG@10', 'Recall@1']
    x = range(len(labels))
    width = 0.35
    algo1 = summary['LensKit.ItemKNNScorer']
    algo2 = summary['LensKit.ImplicitMFScorer']

    plt.figure(figsize=(8, 4))
    plt.bar([i - width / 2 for i in x], [algo1['NDCG@10'] or 0.0, algo1['Recall@1'] or 0.0], width=width, label='ItemKNN')
    plt.bar([i + width / 2 for i in x], [algo2['NDCG@10'] or 0.0, algo2['Recall@1'] or 0.0], width=width, label='ImplicitMF')
    plt.xticks(list(x), labels)
    plt.ylabel('Score')
    plt.title('MovieLens 1M Metrics Comparison')
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    print(f'Saved plot to {plot_path}')


if __name__ == '__main__':
    main()
