import os
import json
import pandas as pd
import matplotlib.pyplot as plt

from omnirec import RecSysDataSet
from omnirec.data_loaders.datasets import DataSet
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.split import UserHoldout
from omnirec.runner.plan import ExperimentPlan
from omnirec.runner.evaluation import Evaluator
from omnirec.metrics.ranking import Recall
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


def main():
    working_dir = os.path.join(os.getcwd(), 'working')
    os.makedirs(working_dir, exist_ok=True)

    set_random_state(42)

    dataset = RecSysDataSet.use_dataloader(DataSet.MovieLens1M)
    print('Loaded dataset:', dataset.meta.name)
    print('Raw interactions:', dataset.num_interactions())

    pipeline = Pipe(
        MakeImplicit(4),
        UserHoldout(validation_size=0.15, test_size=0.15),
    )
    dataset = pipeline.process(dataset)

    train_df = dataset._data.get('train')
    val_df = dataset._data.get('val')
    test_df = dataset._data.get('test')
    stats = {
        'dataset': dataset.meta.name,
        'train_interactions': int(len(train_df)) if train_df is not None else 0,
        'val_interactions': int(len(val_df)) if val_df is not None else 0,
        'test_interactions': int(len(test_df)) if test_df is not None else 0,
        'total_interactions': int(
            (len(train_df) if train_df is not None else 0)
            + (len(val_df) if val_df is not None else 0)
            + (len(test_df) if test_df is not None else 0)
        ),
    }
    print('Post-preprocessing stats:', stats)

    plan = ExperimentPlan(plan_name='movielens1m_prototype')
    plan.add_algorithm('RecBole.Pop', {})

    evaluator = Evaluator(Recall([10]))
    results = run_omnirec(datasets=dataset, plan=plan, evaluator=evaluator)
    print(results)

    results_path = os.path.join(working_dir, 'prototype_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        try:
            json.dump(str(results), f, indent=2)
        except Exception:
            f.write(str(results))

    recall10 = _extract_metric(results, 'Recall@10')
    print({'Recall@10': recall10})

    plot_path = os.path.join(working_dir, 'prototype_metrics.png')
    plt.figure(figsize=(5, 4))
    plt.bar(['Recall@10'], [recall10 if recall10 is not None else 0.0])
    plt.ylabel('Score')
    plt.title('Prototype Metric on MovieLens1M')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    print(f'Saved plot to {plot_path}')


if __name__ == '__main__':
    main()
