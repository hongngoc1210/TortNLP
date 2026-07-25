import argparse
import json
from pathlib import Path


def read_jsonl(path: str) -> list[dict]:
    records = []

    with Path(path).open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, 1):
            line = line.strip()
            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_no}: {error}"
                ) from error

    return records


def to_binary(value, field_name: str, tort_id: str) -> int:
    if isinstance(value, bool):
        return int(value)

    if value in (0, 1):
        return int(value)

    raise ValueError(
        f"{field_name} must be bool/0/1 for tort_id={tort_id!r}; "
        f"received {value!r}"
    )


def binary_f1(
    gold: list[int],
    pred: list[int],
    *,
    empty_value: float = 0.0,
) -> float:
    if len(gold) != len(pred):
        raise ValueError(
            f"F1 input length mismatch: gold={len(gold)}, pred={len(pred)}"
        )

    tp = sum(
        g == 1 and p == 1
        for g, p in zip(gold, pred)
    )
    fp = sum(
        g == 0 and p == 1
        for g, p in zip(gold, pred)
    )
    fn = sum(
        g == 1 and p == 0
        for g, p in zip(gold, pred)
    )

    denominator = 2 * tp + fp + fn

    if denominator == 0:
        return float(empty_value)

    return (2 * tp) / denominator


def safe_mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a TortNLP submission against labeled JSONL. "
            "Reports both per-case macro metrics and global claim-level F1."
        )
    )
    parser.add_argument(
        "--gold",
        required=True,
        help="Labeled JSONL containing court_decision and is_accepted.",
    )
    parser.add_argument(
        "--submission",
        required=True,
        help="JSONL containing tort_id, T_pred, rP_pred, rD_pred.",
    )
    args = parser.parse_args()

    gold_records = read_jsonl(args.gold)
    submission_records = read_jsonl(args.submission)

    gold_by_id = {
        str(record["tort_id"]): record
        for record in gold_records
    }
    pred_by_id = {
        str(record["tort_id"]): record
        for record in submission_records
    }

    if len(gold_by_id) != len(gold_records):
        raise ValueError("Gold file contains duplicate tort_id values.")

    if len(pred_by_id) != len(submission_records):
        raise ValueError("Submission contains duplicate tort_id values.")

    missing = sorted(set(gold_by_id) - set(pred_by_id))
    extra = sorted(set(pred_by_id) - set(gold_by_id))

    if missing:
        raise ValueError(
            "Submission is missing tort_id values: "
            + ", ".join(missing[:20])
        )

    if extra:
        raise ValueError(
            "Submission contains unknown tort_id values: "
            + ", ".join(extra[:20])
        )

    correct_tort = 0

    # Per-case scores: used to diagnose instance-macro evaluation.
    macro_all_empty0: list[float] = []
    macro_all_empty1: list[float] = []
    macro_p_empty0: list[float] = []
    macro_p_empty1: list[float] = []
    macro_d_empty0: list[float] = []
    macro_d_empty1: list[float] = []

    # Global claim arrays: likely closer to many training evaluators.
    global_gold_all: list[int] = []
    global_pred_all: list[int] = []
    global_gold_p: list[int] = []
    global_pred_p: list[int] = []
    global_gold_d: list[int] = []
    global_pred_d: list[int] = []

    no_gold_positive = 0
    both_empty = 0
    total_p_claims = 0
    total_d_claims = 0

    for tort_id, gold in gold_by_id.items():
        pred = pred_by_id[tort_id]

        gold_t = to_binary(
            gold.get("court_decision"),
            "court_decision",
            tort_id,
        )
        pred_t = to_binary(
            pred.get("T_pred"),
            "T_pred",
            tort_id,
        )
        correct_tort += int(gold_t == pred_t)

        gold_p = [
            to_binary(
                claim.get("is_accepted"),
                "plaintiff_claims.is_accepted",
                tort_id,
            )
            for claim in gold.get("plaintiff_claims", [])
        ]
        gold_d = [
            to_binary(
                claim.get("is_accepted"),
                "defendant_claims.is_accepted",
                tort_id,
            )
            for claim in gold.get("defendant_claims", [])
        ]

        pred_p = [
            to_binary(value, "rP_pred", tort_id)
            for value in pred.get("rP_pred", [])
        ]
        pred_d = [
            to_binary(value, "rD_pred", tort_id)
            for value in pred.get("rD_pred", [])
        ]

        if len(gold_p) != len(pred_p):
            raise ValueError(
                f"Plaintiff claim count mismatch for tort_id={tort_id!r}: "
                f"gold={len(gold_p)}, pred={len(pred_p)}"
            )

        if len(gold_d) != len(pred_d):
            raise ValueError(
                f"Defendant claim count mismatch for tort_id={tort_id!r}: "
                f"gold={len(gold_d)}, pred={len(pred_d)}"
            )

        total_p_claims += len(gold_p)
        total_d_claims += len(gold_d)

        gold_all = gold_p + gold_d
        pred_all = pred_p + pred_d

        if sum(gold_all) == 0:
            no_gold_positive += 1

        if sum(gold_all) == 0 and sum(pred_all) == 0:
            both_empty += 1

        macro_all_empty0.append(
            binary_f1(gold_all, pred_all, empty_value=0.0)
        )
        macro_all_empty1.append(
            binary_f1(gold_all, pred_all, empty_value=1.0)
        )

        macro_p_empty0.append(
            binary_f1(gold_p, pred_p, empty_value=0.0)
        )
        macro_p_empty1.append(
            binary_f1(gold_p, pred_p, empty_value=1.0)
        )

        macro_d_empty0.append(
            binary_f1(gold_d, pred_d, empty_value=0.0)
        )
        macro_d_empty1.append(
            binary_f1(gold_d, pred_d, empty_value=1.0)
        )

        global_gold_all.extend(gold_all)
        global_pred_all.extend(pred_all)
        global_gold_p.extend(gold_p)
        global_pred_p.extend(pred_p)
        global_gold_d.extend(gold_d)
        global_pred_d.extend(pred_d)

    n_cases = len(gold_records)
    tort_accuracy = correct_tort / max(n_cases, 1)

    global_f1_all = binary_f1(
        global_gold_all,
        global_pred_all,
        empty_value=0.0,
    )
    global_f1_p = binary_f1(
        global_gold_p,
        global_pred_p,
        empty_value=0.0,
    )
    global_f1_d = binary_f1(
        global_gold_d,
        global_pred_d,
        empty_value=0.0,
    )

    print("=" * 66)
    print("COLIEE LJPJT local evaluation")
    print("=" * 66)
    print(f"Cases                                  : {n_cases}")
    print(f"Plaintiff claims                       : {total_p_claims}")
    print(f"Defendant claims                       : {total_d_claims}")
    print(f"Cases with no gold-positive rationale  : {no_gold_positive}")
    print(f"Cases where gold and prediction empty  : {both_empty}")
    print("-" * 66)
    print(f"Tort Prediction accuracy               : {tort_accuracy:.6f}")
    print("-" * 66)
    print("Per-case macro F1")
    print(f"  All, empty case -> 0                 : {safe_mean(macro_all_empty0):.6f}")
    print(f"  All, empty case -> 1                 : {safe_mean(macro_all_empty1):.6f}")
    print(f"  Plaintiff, empty case -> 0           : {safe_mean(macro_p_empty0):.6f}")
    print(f"  Plaintiff, empty case -> 1           : {safe_mean(macro_p_empty1):.6f}")
    print(f"  Defendant, empty case -> 0           : {safe_mean(macro_d_empty0):.6f}")
    print(f"  Defendant, empty case -> 1           : {safe_mean(macro_d_empty1):.6f}")
    print("-" * 66)
    print("Global claim-level F1")
    print(f"  All claims                           : {global_f1_all:.6f}")
    print(f"  Plaintiff claims                     : {global_f1_p:.6f}")
    print(f"  Defendant claims                     : {global_f1_d:.6f}")
    print("=" * 66)
    print(
        "Do not label either empty-case convention as official until it "
        "is confirmed against the official evaluator."
    )


if __name__ == "__main__":
    main()