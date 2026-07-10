import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from omnirec import RecSysDataSet
from omnirec.data_loaders.datasets import DataSet
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.split import UserHoldout
from omnirec.runner.plan import ExperimentPlan
from omnirec.runner.evaluation import Evaluator
from omnirec.metrics.ranking import NDCG, Recall
from omnirec.runner.algos import LensKit
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state


def main():
    working_dir = os.path.join(os.getcwd(), 'working')
    os.makedirs(working_dir, exist_ok=True)
    set_random_state(42)

    # Load exactly one dataset: MovieLens1M
    dataset = RecSysDataSet.use_dataloader(DataSet.MovieLens1M)

    # Minimal implicit-feedback preprocessing and split
    pipeline = Pipe(
        MakeImplicit(4),
        UserHoldout(0.2, 0.2)
    )
    dataset = pipeline.process(dataset)

    # Exactly one algorithm for prototype validation
    plan = ExperimentPlan(plan_name='MovieLens1M_Prototype')
    plan.add_algorithm(
        LensKit.ItemKNNScorer,
        {
            'max_nbrs': 50,
            'min_nbrs': 1,
            'center': True,
        }
    )

    # Minimal metric subset
    evaluator = Evaluator(
        NDCG([10]),
        Recall([1])
    )

    run_omnirec(datasets=dataset, plan=plan, evaluator=evaluator)

    results = evaluator.get_results()
    dataset_id, df = next(iter(results.items()))
    print('\nRaw results:')
    print(df)

    summary = df.groupby(['algorithm', 'name', 'k'], as_index=False)['value'].mean()
    print('\nSummary:')
    print(summary)

    # One basic plot to demonstrate reporting pipeline
    plot_df = summary.copy()
    plot_df['metric'] = plot_df['name'] + '@' + plot_df['k'].astype(str)

    plt.figure(figsize=(6, 4))
    plt.bar(plot_df['metric'], plot_df['value'])
    plt.ylabel('Value')
    plt.title('MovieLens1M Prototype Metrics')
    plt.tight_layout()
    plot_path = Path(working_dir) / 'prototype_metrics.png'
    plt.savefig(plot_path, dpi=150)
    print(f'Plot saved to: {plot_path}')


if __name__ == '__main__':
    main()
