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


def binary_f1(gold: list[int], pred: list[int]) -> float:
    if len(gold) != len(pred):
        raise ValueError(
            f"F1 input length mismatch: gold={len(gold)}, pred={len(pred)}"
        )

    tp = sum(g == 1 and p == 1 for g, p in zip(gold, pred))
    fp = sum(g == 0 and p == 1 for g, p in zip(gold, pred))
    fn = sum(g == 1 and p == 0 for g, p in zip(gold, pred))

    denominator = 2 * tp + fp + fn
    return 0.0 if denominator == 0 else (2 * tp) / denominator


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a minimal TortNLP submission against labeled JSONL."
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

    duplicate_check = len(pred_by_id) != len(submission_records)
    if duplicate_check:
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
    instance_f1_all: list[float] = []
    instance_f1_p: list[float] = []
    instance_f1_d: list[float] = []

    global_gold: list[int] = []
    global_pred: list[int] = []

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

        gold_all = gold_p + gold_d
        pred_all = pred_p + pred_d

        instance_f1_all.append(binary_f1(gold_all, pred_all))
        instance_f1_p.append(binary_f1(gold_p, pred_p))
        instance_f1_d.append(binary_f1(gold_d, pred_d))

        global_gold.extend(gold_all)
        global_pred.extend(pred_all)

    n_cases = len(gold_records)
    tort_accuracy = correct_tort / max(n_cases, 1)
    rationale_f1_all = sum(instance_f1_all) / max(n_cases, 1)
    rationale_f1_p = sum(instance_f1_p) / max(n_cases, 1)
    rationale_f1_d = sum(instance_f1_d) / max(n_cases, 1)
    rationale_micro_f1 = binary_f1(global_gold, global_pred)

    print("=" * 58)
    print("COLIEE LJPJT local evaluation")
    print("=" * 58)
    print(f"Cases                         : {n_cases}")
    print(f"Tort Prediction accuracy      : {tort_accuracy:.6f}")
    print(f"Rationale F1 All (official)   : {rationale_f1_all:.6f}")
    print(f"Rationale F1 Plaintiff        : {rationale_f1_p:.6f}")
    print(f"Rationale F1 Defendant        : {rationale_f1_d:.6f}")
    print(f"Rationale micro-F1 diagnostic : {rationale_micro_f1:.6f}")
    print("=" * 58)
    print(
        "Note: when a case has no gold-positive and no predicted-positive "
        "claims, this local script assigns F1=0."
    )


if __name__ == "__main__":
    main()