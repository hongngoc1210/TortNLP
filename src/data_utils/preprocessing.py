import json


def load_jsonl(path):

    samples = []

    with open(path, "r", encoding="utf-8") as f:

        for line in f:
            line = line.strip()

            if not line:
                continue

            samples.append(json.loads(line))

    return samples


def normalize_case(case):

    # ------------------------------
    # undisputed facts
    # ------------------------------

    facts = []

    for f in case.get("undisputed_facts", []):
        facts.append(f.get("description", ""))

    U_text = " ".join(facts)

    # ------------------------------
    # plaintiff claims
    # ------------------------------

    P = []
    R_P = []

    for c in case.get("plaintiff_claims", []):

        P.append(c.get("description", ""))

        label = c.get("is_accepted")

        # None appears in test set
        if label is None:
            R_P.append(-1)
        else:
            R_P.append(int(label))

    # ------------------------------
    # defendant claims
    # ------------------------------

    D = []
    R_D = []

    for c in case.get("defendant_claims", []):

        D.append(c.get("description", ""))

        label = c.get("is_accepted")

        if label is None:
            R_D.append(-1)
        else:
            R_D.append(int(label))

    # ------------------------------
    # court decision
    # ------------------------------

    T = case.get("court_decision")

    if T is None:
        T = -1
    else:
        T = int(T)

    # ------------------------------

    return {

        "tort_id": case.get("tort_id", ""),

        "U": U_text,

        "P": P,
        "D": D,

        "R_P": R_P,
        "R_D": R_D,

        "T": T,
    }


def build_dataset(jsonl_path):

    raw = load_jsonl(jsonl_path)

    dataset = []

    for case in raw:

        try:

            dataset.append(
                normalize_case(case)
            )

        except Exception as e:

            print("Skip corrupted case:", e)

    return dataset