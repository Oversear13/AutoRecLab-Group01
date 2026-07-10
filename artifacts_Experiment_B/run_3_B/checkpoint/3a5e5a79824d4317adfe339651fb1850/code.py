import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from omnirec import RecSysDataSet, NDCG, Recall
from omnirec.data_loaders.datasets import DataSet
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.split import UserHoldout
from omnirec.runner.evaluation import Evaluator
from omnirec.runner.plan import ExperimentPlan
from omnirec.runner.algos import LensKit
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state


def build_dataset():
    dataset = RecSysDataSet.use_dataloader(DataSet.MovieLens1M)
    print('Loaded dataset')
    print(dataset)
    print('Interactions:', dataset.num_interactions())
    if hasattr(dataset, 'min_rating'):
        try:
            print('Min rating:', dataset.min_rating())
            print('Max rating:', dataset.max_rating())
        except Exception:
            pass

    pipe = Pipe(
        MakeImplicit(4),
        CorePruning(5),
        UserHoldout(0.8, 0.2),
    )
    dataset = pipe.process(dataset)
    print('Preprocessed dataset')
    print(dataset)
    return dataset


def build_plan():
    plan = ExperimentPlan(plan_name='MovieLens1M_Prototype')
    plan.add_algorithm(LensKit.ItemKNNScorer, {'max_nbrs': 20, 'min_nbrs': 5})
    return plan


def main():
    working_dir = os.path.join(os.getcwd(), 'working')
    os.makedirs(working_dir, exist_ok=True)
    os.chdir(working_dir)

    set_random_state(42)
    dataset = build_dataset()
    plan = build_plan()
    evaluator = Evaluator(NDCG([10]), Recall([1]))

    run_omnirec(datasets=dataset, plan=plan, evaluator=evaluator)

    results_by_dataset = evaluator.get_results()
    for dataset_id, df in results_by_dataset.items():
        print('\nResults for:', dataset_id)
        print(df)
        summary = df.groupby(['algorithm', 'name', 'k'], as_index=False)['value'].mean()
        print('\nSummary:')
        print(summary)

        out_csv = Path('evaluation_results.csv')
        df.to_csv(out_csv, index=False)

        pivot = summary.pivot(index='algorithm', columns=['name', 'k'], values='value')
        ax = pivot.plot(kind='bar', figsize=(8, 4))
        ax.set_ylabel('Metric value')
        ax.set_title('MovieLens1M Prototype Results')
        plt.tight_layout()
        plot_path = Path('prototype_metrics.png')
        plt.savefig(plot_path, dpi=150)
        print(f'Saved plot to {plot_path.resolve()}')
        print(f'Saved results to {out_csv.resolve()}')


if __name__ == '__main__':
    main()
