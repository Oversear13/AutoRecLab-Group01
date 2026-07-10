import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from omnirec import NDCG, Recall, RecSysDataSet
from omnirec.data_loaders.datasets import DataSet
from omnirec.preprocess.core_pruning import CorePruning
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
        'dataset_name': getattr(getattr(dataset, 'meta', None), 'name', 'MovieLens1M'),
        'num_interactions': int(dataset.num_interactions()),
    }

    pipeline = Pipe(
        MakeImplicit(4),
        CorePruning(5),
        UserHoldout(validation_size=0.05, test_size=0.05),
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
    plan.add_algorithm(
        LensKit.ImplicitMFScorer,
        {
            'n_factors': 50,
            'n_iters': 20,
            'reg': 0.1,
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

    comparison_df = summary_df.copy()
    comparison_df['metric'] = comparison_df['name'] + '@' + comparison_df['k'].astype(str)
    comparison_pivot = comparison_df.pivot(index='algorithm', columns='metric', values='value').reset_index()
    print('\nComparison:')
    print(comparison_pivot.to_json(orient='records'))

    metrics = ['NDCG@10', 'Recall@1']
    algos = comparison_pivot['algorithm'].tolist()
    x = range(len(metrics))
    width = 0.35

    plt.figure(figsize=(7, 4))
    for i, algo in enumerate(algos[:2]):
        values = [float(comparison_pivot.loc[comparison_pivot['algorithm'] == algo, m].iloc[0]) for m in metrics]
        plt.bar([j + (i - 0.5) * width for j in x], values, width=width, label=algo)

    plt.xticks(list(x), metrics)
    plt.ylabel('Value')
    plt.title('MovieLens1M Comparison: NDCG@10 and Recall@1')
    plt.legend()
    plt.tight_layout()
    plot_path = Path(working_dir) / 'comparison_metrics.png'
    plt.savefig(plot_path, dpi=150)
    print(f'Plot saved to: {plot_path}')


if __name__ == '__main__':
    main()
