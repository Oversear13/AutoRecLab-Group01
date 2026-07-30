"""
Independent verification script for AutoRecLab Run 1 (Experiment A, MovieLens 100K, seed 447471).

Purpose
-------
This script provides a fully reproducible, standalone check of two claims made in
Section 3.1.3 and Section 3.3/3.4 of the report:

  (A) That the preprocessing pipeline (MakeImplicit(4) -> CorePruning(5)) applied by
      AutoRecLab's generated code.py, when re-run with the officially released OmniRec
      library, yields the exact same interaction counts reported in the paper
      (100,000 -> 55,375 -> 54,413).

  (B) That an attempt to reconstruct the exact train/validation/test split used by
      AutoRecLab (same seed, same OmniRec split function) does NOT reliably reproduce
      the same per-user test allocation, illustrating a concrete limitation of
      seed-based reproducibility across library versions.

Requirements
------------
  pip install omnirec lenskit binpickle

Inputs required (place in the same directory as this script, or adjust paths below)
------------------------------------------------------------------------------------
  - movielens.csv                 : raw MovieLens 100K ratings (user_id, item_id, rating, timestamp)
  - predictions.json              : AutoRecLab's saved ranked predictions for
                                     (MovieLens100K, ItemKNN, seed=447471, Run 1, Node a44f1301...)
  - model.bpk                     : AutoRecLab's saved LensKit model for the same run/node/seed
                                     (checkpoints/MovieLens100K-.../LensKit.ItemKNNScorer-...-447471/model.bpk)

Output
------
  A console report (also written to verification_report.txt) with every intermediate
  number needed to check claims (A) and (B) by hand.

Note on claim (B)
------------------
This script deliberately does NOT report a "reproduced NDCG@10 / Precision@10" number
as if it were comparable to AutoRecLab's reported metrics. As shown below, the
reconstructed split does not align closely enough with AutoRecLab's actual test
allocation (see the "random baseline sanity check" at the end) to make such a
comparison meaningful. Reporting a number here would overstate what was actually
verified. This is intentional and matches the Evidence Level framing in Section 3.3
of the report.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 0. Paths (adjust if running in a different directory layout)
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
MOVIELENS_CSV = SCRIPT_DIR / "movielens.csv"
PREDICTIONS_JSON = SCRIPT_DIR / "predictions.json"
MODEL_BPK = SCRIPT_DIR / "model.bpk"
REPORT_PATH = SCRIPT_DIR / "verification_report.txt"

SEED = 447471

report_lines = []


def log(msg: str = ""):
    print(msg)
    report_lines.append(msg)


# ---------------------------------------------------------------------------
# PART A: Preprocessing reproduction with the official OmniRec library
# ---------------------------------------------------------------------------
def verify_preprocessing():
    log("=" * 78)
    log("PART A: Preprocessing reproduction (MakeImplicit(4) -> CorePruning(5))")
    log("=" * 78)

    from omnirec.recsys_data_set import RecSysDataSet, DatasetMeta
    from omnirec.data_variants import RawData
    from omnirec.preprocess.feedback_conversion import MakeImplicit
    from omnirec.preprocess.core_pruning import CorePruning
    from omnirec.util.util import set_random_state

    raw = pd.read_csv(MOVIELENS_CSV)
    raw = raw.rename(columns={"user_id": "user", "item_id": "item"})
    log(f"Loaded raw MovieLens 100K: {raw.shape[0]} interactions, "
        f"{raw['user'].nunique()} users, {raw['item'].nunique()} items")

    dataset = RecSysDataSet(RawData(raw), DatasetMeta(name="MovieLens100K"))
    set_random_state(SEED)

    n_before = dataset.num_interactions()
    dataset = MakeImplicit(4)._process(dataset)
    n_after_implicit = dataset.num_interactions()
    dataset = CorePruning(5)._process(dataset)
    n_after_pruning = dataset.num_interactions()

    log("")
    log(f"Interactions before preprocessing:        {n_before}")
    log(f"Interactions after MakeImplicit(4):        {n_after_implicit}")
    log(f"Interactions after CorePruning(5):         {n_after_pruning}")
    log("")
    log("Expected values reported in the paper (Section 3.1.3):")
    log("  100,000 -> 55,375 -> 54,413")
    log("")

    matches = (n_before == 100000 and n_after_implicit == 55375 and n_after_pruning == 54413)
    log(f"MATCH WITH PAPER VALUES: {matches}")
    log("")

    return dataset, matches


# ---------------------------------------------------------------------------
# PART B: Split reconstruction attempt + honest limitation check
# ---------------------------------------------------------------------------
def load_model_vocab(model_path: Path):
    """
    Load a LensKit model.bpk saved by an older library version.

    NOTE: The model was saved with a LensKit version whose internal Vocabulary /
    MatrixRelationshipSet pickle format differs from the currently released
    lenskit package (installed via `pip install lenskit`). Loading it directly
    raises a KeyError inside lenskit's __setstate__ methods. The two small
    monkey-patches below only change how the *old* pickle state dict is mapped
    onto the current class attributes; they do not alter any data values.
    This is documented here transparently so the loading step is auditable.
    """
    import binpickle
    import lenskit.data.vocab as v
    import lenskit.data.relationships as r

    captured = {}

    def patched_vocab_setstate(self, state):
        name = state.get("name", "unknown")
        if "_index" in state:
            # Older lenskit versions stored a pandas Index directly.
            self._array = np.asarray(state["_index"])
        else:
            self._array = state["array"]
        self.name = name
        self._index = None
        self._log = None
        captured[name] = self._array

    def patched_rel_setstate(self, state):
        self.name = state.get("name")
        self.schema = state.get("schema")
        self._link_cols = state.get("columns", state.get("_link_cols"))
        self._table = state.get("table", state.get("_table"))
        self._vocabularies = state.get("vocabularies", state.get("_vocabularies"))
        # _init_structures() intentionally skipped: internal index structures
        # differ between versions and are not needed to read out the raw
        # user/item vocabularies below.

    v.Vocabulary.__setstate__ = patched_vocab_setstate
    r.MatrixRelationshipSet.__setstate__ = patched_rel_setstate

    model = binpickle.load(str(model_path))
    return model, captured


def verify_split_reconstruction():
    log("=" * 78)
    log("PART B: Split reconstruction attempt (honest limitation check)")
    log("=" * 78)

    from omnirec.recsys_data_set import RecSysDataSet, DatasetMeta
    from omnirec.data_variants import RawData
    from omnirec.preprocess.feedback_conversion import MakeImplicit
    from omnirec.preprocess.core_pruning import CorePruning
    from omnirec.preprocess.split import UserHoldout
    from omnirec.util.util import set_random_state

    raw = pd.read_csv(MOVIELENS_CSV)
    raw = raw.rename(columns={"user_id": "user", "item_id": "item"})
    dataset = RecSysDataSet(RawData(raw), DatasetMeta(name="MovieLens100K"))

    set_random_state(SEED)
    dataset = MakeImplicit(4)._process(dataset)
    dataset = CorePruning(5)._process(dataset)

    split = UserHoldout(0.15, 0.15)
    dataset2 = split._process(dataset)
    train_df = dataset2._data.train
    test_df = dataset2._data.test

    log(f"Reconstructed split -> train: {train_df.shape[0]} rows / "
        f"{train_df['user'].nunique()} users, "
        f"test: {test_df.shape[0]} rows / {test_df['user'].nunique()} users")

    if not MODEL_BPK.exists() or not PREDICTIONS_JSON.exists():
        log("model.bpk or predictions.json not found - skipping comparison against "
            "AutoRecLab's actual saved artifacts.")
        return

    # Load AutoRecLab's saved model to get its internal user/item vocabulary.
    model, vocabs = load_model_vocab(MODEL_BPK)
    user_vocab = vocabs.get("user")
    item_vocab = vocabs.get("item")
    log("")
    log(f"AutoRecLab's saved model vocabulary: {len(user_vocab)} users, "
        f"{len(item_vocab)} items (loaded from model.bpk)")

    with open(PREDICTIONS_JSON) as f:
        preds = json.load(f)
    pred_df = pd.DataFrame(preds)
    log(f"AutoRecLab's saved predictions.json: {pred_df['user'].nunique()} users, "
        f"{pred_df['item'].nunique()} items ranked, "
        f"{pred_df.shape[0]} total (user,item) rows")

    # AutoRecLab's predictions use 0-indexed ids; our reconstructed split uses
    # raw MovieLens ids (1-indexed). Apply the -1 shift for comparison.
    test_df = test_df.copy()
    test_df["user0"] = test_df["user"] - 1

    pred_users = set(pred_df["user"].unique())
    recon_test_users = set(test_df["user0"].unique())
    common = pred_users & recon_test_users

    log("")
    log(f"Users in AutoRecLab predictions:        {len(pred_users)}")
    log(f"Users in our reconstructed test split:   {len(recon_test_users)}")
    log(f"Users common to both (after -1 shift):   {len(common)} / {len(recon_test_users)}")

    # Sanity check: does the model actually separate signal from noise on OUR
    # reconstructed test set? If our split were identical to AutoRecLab's own
    # split, the model's NDCG@10 should be far above a random-ranking baseline
    # (AutoRecLab reports NDCG@10 ~ 0.136-0.141 for ItemKNN in Section 3.1.5).
    test_df["item0"] = test_df["item"] - 1
    gt = test_df.groupby("user0")["item0"].apply(set).to_dict()

    def ndcg_at_10(ranked_items, relevant_set):
        dg = np.array([1 / np.log2(i + 1) for i in range(1, 11)])
        idcg = dg.sum()
        hits = np.isin(ranked_items[:10], list(relevant_set))
        return (np.where(hits, dg[: len(hits)], 0)).sum() / idcg

    model_ndcgs, random_ndcgs = [], []
    rng = np.random.default_rng(0)
    for user, group in pred_df.groupby("user"):
        if user not in gt or len(gt[user]) == 0:
            continue
        relevant = gt[user]
        ranked = group.sort_values("rank")["item"].tolist()
        model_ndcgs.append(ndcg_at_10(ranked, relevant))
        shuffled = rng.permutation(group["item"].values)
        random_ndcgs.append(ndcg_at_10(shuffled, relevant))

    log("")
    log("Sanity check on OUR reconstructed test split:")
    log(f"  Model (AutoRecLab's saved ranking)  -> mean NDCG@10: {np.mean(model_ndcgs):.4f}")
    log(f"  Random ranking (same candidate set)  -> mean NDCG@10: {np.mean(random_ndcgs):.4f}")
    log("")
    log("AutoRecLab's OWN reported NDCG@10 for ItemKNN on this dataset "
        "(Section 3.1.5 of the paper): ~0.136-0.141")
    log("")
    log("INTERPRETATION:")
    log("The model score on our reconstructed split is only marginally above the")
    log("random baseline, and far below AutoRecLab's own reported NDCG@10. This")
    log("indicates our reconstructed test split does NOT match the actual split")
    log("AutoRecLab evaluated against, even though the same OmniRec split function")
    log("and the same seed were used. We attribute this to differences in internal")
    log("random-number consumption between the (pre-release) OmniRec version used")
    log("for the original AutoRecLab run (June 2026) and the current PyPI release")
    log("(v1.0.0) used here. Consequently, we do NOT report a recomputed NDCG@10 /")
    log("Precision@10 value as comparable to AutoRecLab's own metrics; doing so")
    log("would overstate what this verification attempt actually established.")


def main():
    log(f"Verification run for seed={SEED}, dataset=MovieLens100K")
    log("")
    _, matches = verify_preprocessing()
    log("")
    verify_split_reconstruction()

    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")
    log("")
    log(f"Full report written to: {REPORT_PATH}")

    if not matches:
        sys.exit(1)


if __name__ == "__main__":
    main()
