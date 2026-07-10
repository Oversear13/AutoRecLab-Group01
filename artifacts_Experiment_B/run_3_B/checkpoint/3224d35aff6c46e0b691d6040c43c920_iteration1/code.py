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


def _get_metric_value(results, metric_name):
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

    results_path = os.path.join(working_dir, 'prototype_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        try:
            json.dump(str(results), f, indent=2)
        except Exception:
            f.write(str(results))

    ndcg10 = _get_metric_value(results, 'NDCG@10')
    recall1 = _get_metric_value(results, 'Recall@1')
    print({'NDCG@10': ndcg10, 'Recall@1': recall1})

    plot_path = os.path.join(working_dir, 'prototype_metrics.png')
    plt.figure(figsize=(7, 4))
    labels = ['NDCG@10', 'Recall@1']
    values = [ndcg10 if ndcg10 is not None else 0.0, recall1 if recall1 is not None else 0.0]
    plt.bar(labels, values)
    plt.ylabel('Score')
    plt.title('MovieLens 1M Metrics')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    print(f'Saved plot to {plot_path}')


if __name__ == '__main__':
    main()
