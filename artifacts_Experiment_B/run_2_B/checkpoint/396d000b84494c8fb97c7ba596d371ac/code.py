import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from omnirec import RecSysDataSet, NDCG, Recall
from omnirec.data_loaders.datasets import DataSet
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.split import UserHoldout
from omnirec.runner.algos import LensKit
from omnirec.runner.evaluation import Evaluator
from omnirec.runner.plan import ExperimentPlan
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state


def main():
    working_dir = os.path.join(os.getcwd(), 'working')
    os.makedirs(working_dir, exist_ok=True)

    set_random_state(42)

    dataset = RecSysDataSet.use_dataloader(DataSet.MovieLens1M)
    dataset_summary = {
        'dataset_name': getattr(dataset.meta, 'name', 'MovieLens1M'),
        'num_interactions': int(dataset.num_interactions()),
    }

    pipeline = Pipe(
        MakeImplicit(4),
        UserHoldout(0.1, 0.1),
    )
    dataset = pipeline.process(dataset)

    plan = ExperimentPlan(plan_name='MovieLens1M_Prototype')
    plan.add_algorithm(
        LensKit.ItemKNNScorer,
        {
            'max_nbrs': 50,
            'min_nbrs': 1,
            'center': True,
        }
    )

    evaluator = Evaluator(
        NDCG([10]),
        Recall([1]),
    )

    run_omnirec(dataset, plan, evaluator)

    results = evaluator.get_results()
    dataset_id, df = next(iter(results.items()))

    print('\nDataset summary:')
    print(pd.Series(dataset_summary).to_json())

    print('\nRaw results:')
    print(df.to_json(orient='records'))

    summary_df = df.groupby(['algorithm', 'name', 'k'], as_index=False)['value'].mean()
    print('\nSummary:')
    print(summary_df.to_json(orient='records'))

    plot_df = summary_df.copy()
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
