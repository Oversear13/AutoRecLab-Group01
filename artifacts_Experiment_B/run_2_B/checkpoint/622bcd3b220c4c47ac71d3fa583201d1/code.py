import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from omnirec import NDCG, Recall, RecSysDataSet
from omnirec.data_loaders.datasets import DataSet
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.split import RandomHoldout
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

    pipeline = Pipe(
        MakeImplicit(4),
        RandomHoldout(validation_size=0.1, test_size=0.1),
    )
    dataset = pipeline.process(dataset)

    plan = ExperimentPlan(plan_name='MovieLens1M_Prototype')
    plan.add_algorithm(
        LensKit.ItemKNNScorer,
        {
            'max_nbrs': 50,
            'min_nbrs': 1,
            'center': True,
        },
    )

    evaluator = Evaluator(
        NDCG([10]),
        Recall([1]),
    )

    run_omnirec(datasets=dataset, plan=plan, evaluator=evaluator)

    results = evaluator.get_results()
    dataset_id, df = next(iter(results.items()))

    print('\nDataset summary:')
    summary = {
        'dataset_id': dataset_id,
        'num_rows': int(len(df)),
        'columns': list(df.columns),
        'algorithms': sorted(df['algorithm'].unique().tolist()),
        'metrics': sorted(df['name'].unique().tolist()),
    }
    print(summary)

    print('\nRaw results:')
    print(df.to_string(index=False))

    summary_df = df.groupby(['algorithm', 'name', 'k'], as_index=False)['value'].mean()
    print('\nAggregated results:')
    print(summary_df.to_string(index=False))

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
