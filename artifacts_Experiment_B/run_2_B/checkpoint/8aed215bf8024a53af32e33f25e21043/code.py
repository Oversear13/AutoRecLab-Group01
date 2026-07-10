import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from omnirec import RecSysDataSet
from omnirec.data_loaders.datasets import DataSet
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.split import UserHoldout
from omnirec.runner.plan import ExperimentPlan
from omnirec.runner.algos import RecBole
from omnirec.runner.evaluation import Evaluator
from omnirec.metrics.ranking import NDCG, Recall
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state


def main():
    working_dir = os.path.join(os.getcwd(), 'working')
    os.makedirs(working_dir, exist_ok=True)

    set_random_state(42)

    dataset = RecSysDataSet.use_dataloader(DataSet.MovieLens1M)
    print('Loaded dataset:')
    print(dataset)
    print('Num interactions:', dataset.num_interactions())
    print('Rating range:', dataset.min_rating(), dataset.max_rating())

    pipeline = Pipe(
        MakeImplicit(4),
        CorePruning(5),
        UserHoldout(validation_size=0.1, test_size=0.1),
    )
    dataset = pipeline.process(dataset)

    plan = ExperimentPlan('MovieLens1M_Prototype')
    plan.add_algorithm(RecBole.Pop)

    evaluator = Evaluator(
        NDCG([10]),
        Recall([1]),
    )

    run_omnirec(datasets=dataset, plan=plan, evaluator=evaluator)

    results = evaluator.get_results()
    if not results:
        raise RuntimeError('No results were produced by the evaluation pipeline.')

    dataset_id, df = next(iter(results.items()))
    print('\nEvaluation results for:', dataset_id)
    print(df)

    # Machine-readable output
    raw_results_path = Path(working_dir) / 'evaluation_results.csv'
    df.to_csv(raw_results_path, index=False)
    print('\nSaved raw results to:', raw_results_path)

    summary = df.groupby(['algorithm', 'name', 'k'], as_index=False)['value'].mean()
    summary_path = Path(working_dir) / 'metrics_summary.csv'
    summary.to_csv(summary_path, index=False)
    print('Saved summary to:', summary_path)

    # Basic plot
    plot_df = summary.copy()
    plot_df['metric'] = plot_df.apply(
        lambda r: f"{r['name']}@{int(r['k'])}" if pd.notna(r['k']) else r['name'],
        axis=1,
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(plot_df['metric'], plot_df['value'], color=['steelblue', 'darkorange'])
    ax.set_title('Prototype Ranking Metrics on MovieLens1M')
    ax.set_ylabel('Score')
    ax.set_xlabel('Metric')
    ax.set_ylim(0, max(1.0, float(plot_df['value'].max()) * 1.15))
    for idx, val in enumerate(plot_df['value']):
        ax.text(idx, val, f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plot_path = Path(working_dir) / 'metrics_plot.png'
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    print('Saved plot to:', plot_path)


if __name__ == '__main__':
    main()
