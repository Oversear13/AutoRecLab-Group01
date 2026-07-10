import os
import json
from pathlib import Path
import matplotlib.pyplot as plt

from omnirec import RecSysDataSet, NDCG, Recall
from omnirec.data_loaders.datasets import DataSet
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.split import UserHoldout
from omnirec.runner.plan import ExperimentPlan
from omnirec.runner.algos import LensKit
from omnirec.runner.evaluation import Evaluator
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state


def main():
    working_dir = os.path.join(os.getcwd(), 'working')
    os.makedirs(working_dir, exist_ok=True)
    set_random_state(42)

    dataset = RecSysDataSet.use_dataloader(DataSet.MovieLens1M)
    pipeline = Pipe(
        MakeImplicit(4),
        UserHoldout(validation_size=0.15, test_size=0.15),
    )
    dataset = pipeline.process(dataset)

    plan = ExperimentPlan(plan_name='movielens1m_prototype')
    plan.add_algorithm(LensKit.PopScorer, {})

    evaluator = Evaluator(NDCG([10]), Recall([1]))

    results = run_omnirec(datasets=dataset, plan=plan, evaluator=evaluator)
    print(results)

    results_path = os.path.join(working_dir, 'prototype_results.json')
    try:
        if hasattr(results, 'to_json'):
            with open(results_path, 'w', encoding='utf-8') as f:
                f.write(results.to_json())
        else:
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(str(results), f, indent=2)
    except Exception as e:
        print(f'Could not save structured results: {e}')

    plot_path = os.path.join(working_dir, 'prototype_metrics.png')
    metric_names = ['NDCG@10', 'Recall@1']
    metric_values = [None, None]
    try:
        if hasattr(results, 'iloc'):
            row = results.iloc[0]
            for i, key in enumerate(['NDCG@10', 'Recall@1']):
                if key in results.columns:
                    metric_values[i] = float(row[key])
    except Exception:
        pass

    if any(v is None for v in metric_values):
        metric_values = [0.0, 0.0]
    else:
        metric_values = [float(v) for v in metric_values if v is not None]

    plt.figure(figsize=(6, 4))
    plt.bar(metric_names, metric_values)
    plt.ylabel('Score')
    plt.title('Prototype Top-N Metrics')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    print(f'Saved plot to {plot_path}')


if __name__ == '__main__':
    main()
