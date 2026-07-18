import argparse
import json
from typing import Dict, List

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score


def load_jsonl(path: str) -> List[dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def bool_to_int(x):
    if x is None:
        return -1
    return int(bool(x))


def get_tort_id(case: dict) -> str:
    return str(case.get("tort_id", case.get("case_id", "")))


def get_gold_tp(case: dict) -> int:
    return bool_to_int(case.get("court_decision"))


def get_pred_tp(pred: dict) -> int:
    if "court_decision_label" in pred:
        return int(pred["court_decision_label"])
    if "court_decision" in pred:
        return bool_to_int(pred["court_decision"])
    raise ValueError(f"Prediction missing court_decision: {pred.get('tort_id')}")


def get_gold_re(case: dict):
    labels = []
    meta = []

    for i, c in enumerate(case.get("plaintiff_claims", [])):
        y = c.get("is_accepted")
        if y is not None:
            labels.append(bool_to_int(y))
            meta.append(("P", i))

    for i, c in enumerate(case.get("defendant_claims", [])):
        y = c.get("is_accepted")
        if y is not None:
            labels.append(bool_to_int(y))
            meta.append(("D", i))

    return labels, meta


def get_pred_re(pred: dict, meta):
    labels = []

    accepted_p = set(pred.get("accepted_plaintiff_claims", []))
    accepted_d = set(pred.get("accepted_defendant_claims", []))

    p_claims = pred.get("plaintiff_claims", [])
    d_claims = pred.get("defendant_claims", [])

    for side, idx in meta:
        if side == "P":
            if idx < len(p_claims) and "is_accepted" in p_claims[idx]:
                labels.append(bool_to_int(p_claims[idx]["is_accepted"]))
            else:
                labels.append(1 if f"P{idx}" in accepted_p else 0)

        elif side == "D":
            if idx < len(d_claims) and "is_accepted" in d_claims[idx]:
                labels.append(bool_to_int(d_claims[idx]["is_accepted"]))
            else:
                labels.append(1 if f"D{idx}" in accepted_d else 0)

    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold_path", type=str, required=True)
    parser.add_argument("--pred_path", type=str, required=True)
    args = parser.parse_args()

    gold_cases = load_jsonl(args.gold_path)
    pred_cases = load_jsonl(args.pred_path)

    pred_by_id: Dict[str, dict] = {
        get_tort_id(x): x for x in pred_cases
    }

    tp_gold = []
    tp_pred = []

    re_gold = []
    re_pred = []

    missing_preds = []

    for case in gold_cases:
        tort_id = get_tort_id(case)

        if tort_id not in pred_by_id:
            missing_preds.append(tort_id)
            continue

        pred = pred_by_id[tort_id]

        # TP
        y_tp = get_gold_tp(case)
        if y_tp >= 0:
            tp_gold.append(y_tp)
            tp_pred.append(get_pred_tp(pred))

        # RE
        y_re, meta = get_gold_re(case)
        p_re = get_pred_re(pred, meta)

        if len(y_re) != len(p_re):
            raise ValueError(
                f"RE length mismatch at {tort_id}: gold={len(y_re)}, pred={len(p_re)}"
            )

        re_gold.extend(y_re)
        re_pred.extend(p_re)

    print("========== Evaluation ==========")
    print(f"Gold cases:       {len(gold_cases)}")
    print(f"Prediction cases: {len(pred_cases)}")
    print(f"Matched cases:    {len(gold_cases) - len(missing_preds)}")
    print(f"Missing preds:    {len(missing_preds)}")

    if missing_preds:
        print("First missing ids:", missing_preds[:10])

    print("\n========== TP: Tort Prediction ==========")
    if tp_gold:
        print(f"TP samples:  {len(tp_gold)}")
        print(f"Accuracy:    {accuracy_score(tp_gold, tp_pred):.4f}")
        print(f"F1:          {f1_score(tp_gold, tp_pred, zero_division=0):.4f}")
    else:
        print("No TP gold labels found.")

    print("\n========== RE: Rationale Extraction ==========")
    if re_gold:
        precision, recall, f1, _ = precision_recall_fscore_support(
            re_gold,
            re_pred,
            average="binary",
            zero_division=0,
        )
        print(f"RE claims:   {len(re_gold)}")
        print(f"Accuracy:    {accuracy_score(re_gold, re_pred):.4f}")
        print(f"Precision:   {precision:.4f}")
        print(f"Recall:      {recall:.4f}")
        print(f"F1:          {f1:.4f}")

        print("\nGold label counts:", np.bincount(np.array(re_gold), minlength=2).tolist())
        print("Pred label counts:", np.bincount(np.array(re_pred), minlength=2).tolist())
    else:
        print("No RE gold labels found.")


if __name__ == "__main__":
    main()