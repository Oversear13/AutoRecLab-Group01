from pathlib import Path
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
from omnirec.runner.algos import RecBole
from omnirec.runner.evaluation import Evaluator
from omnirec.metrics.ranking import NDCG, Recall
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state


working_dir = os.path.join(os.getcwd(), 'working')
os.makedirs(working_dir, exist_ok=True)


def main():
    set_random_state(42)

    print('Loading MovieLens20M...')
    dataset = RecSysDataSet.use_dataloader(DataSet.MovieLens20M)
    print('Loaded dataset summary:')
    print(dataset)
    print('Total interactions:', dataset.num_interactions())
    try:
        print('Min rating:', dataset.min_rating())
        print('Max rating:', dataset.max_rating())
    except Exception:
        pass

    pipeline = Pipe(
        MakeImplicit(4),
        UserHoldout(0.15, 0.15),
    )
    dataset = pipeline.process(dataset)

    print('Preprocessing complete.')
    print(dataset.format_details())

    plan = ExperimentPlan('MovieLens20M_Prototype')
    plan.add_algorithm(RecBole.Pop)

    evaluator = Evaluator(NDCG(10), Recall(1))

    print('Running OmniRec experiment...')
    results = run_omnirec(datasets=dataset, plan=plan, evaluator=evaluator)

    print('Experiment finished.')
    print(results)

    metric_rows = []
    if hasattr(results, 'to_dict'):
        try:
            metric_rows = results.to_dict(orient='records')
        except Exception:
            pass
    elif isinstance(results, dict):
        metric_rows = [results]

    if metric_rows:
        out_json = Path(working_dir) / 'results.json'
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump(metric_rows, f, indent=2, default=str)

    ndcg_val = None
    recall_val = None
    if isinstance(results, pd.DataFrame):
        cols = list(results.columns)
        for c in cols:
            cu = c.upper()
            if 'NDCG' in cu and ('10' in cu or '@10' in cu):
                ndcg_val = float(results.iloc[0][c])
            if 'RECALL' in cu and ('1' in cu or '@1' in cu):
                recall_val = float(results.iloc[0][c])
    if ndcg_val is not None and recall_val is not None:
        plt.figure(figsize=(6, 4))
        plt.bar(['NDCG@10', 'Recall@1'], [ndcg_val, recall_val], color=['#4C78A8', '#F58518'])
        plt.ylabel('Score')
        plt.title('MovieLens20M Prototype Metrics')
        plt.tight_layout()
        plot_path = Path(working_dir) / 'prototype_metrics.png'
        plt.savefig(plot_path, dpi=150)
        print(f'Saved plot to {plot_path}')
    else:
        print('Could not parse metric values for plotting; skipping plot generation.')


if __name__ == '__main__':
    main()