import json
import glob
import os


def fuse(ml, nlp, cv, w_ml, w_nlp, w_cv):
    scores = {"ml": ml, "nlp": nlp}
    weights = {"ml": w_ml, "nlp": w_nlp}
    if cv is not None:
        scores["cv"] = cv
        weights["cv"] = w_cv
    total_w = sum(weights.values())
    norm_w = {}
    for k, v in weights.items():
        norm_w[k] = v / total_w
    result = 0
    for k in scores:
        result += scores[k] * norm_w[k]
    return result


def band(score):
    if score < 0.35:
        return "Low"
    elif score < 0.65:
        return "Moderate"
    else:
        return "High"


CONFIGS = {
    "Current": (0.50, 0.30, 0.20),
    "Equal": (0.333, 0.333, 0.333),
    "ML-heavy": (0.60, 0.20, 0.20),
    "NLP-heavy": (0.30, 0.50, 0.20),
    "CV-heavy": (0.30, 0.20, 0.50),
}


def load_assessments(folder="outputs/assessments"):
    files = sorted(glob.glob(os.path.join(folder, "assessment_*.json")))
    cases = []
    for path in files:
        f = open(path, "r", encoding="utf-8")
        data = json.load(f)
        f.close()

        ml = data.get("ml_result", {}).get("ml_score")
        nlp = data.get("nlp_result", {}).get("nlp_score")
        cv_result = data.get("cv_result", {}) or {}
        cv = None
        if cv_result.get("cv_available"):
            cv = cv_result.get("cv_score")

        if ml is None or nlp is None:
            continue

        case = {}
        case["id"] = data.get("assessment_id", "?")[:8]
        case["ml"] = ml
        case["nlp"] = nlp
        case["cv"] = cv
        cases.append(case)
    return cases


def main():
    cases = load_assessments()

    if len(cases) == 0:
        print("No assessment JSON files found in outputs/assessments/.")
        print("Run a few assessments through your app first, then re-run this script.")
        return

    print("Loaded", len(cases), "saved assessments.")
    print("")

    changed_counts = {}
    for name in CONFIGS:
        changed_counts[name] = 0

    baseline_name = "Current"

    for c in cases:
        row = c["id"] + "  "
        bands = {}
        for name in CONFIGS:
            w1, w2, w3 = CONFIGS[name]
            score = fuse(c["ml"], c["nlp"], c["cv"], w1, w2, w3)
            b = band(score)
            bands[name] = b
            row = row + name + "=" + str(round(score, 3)) + "(" + b[0] + ")  "
        print(row)

        for name in CONFIGS:
            if bands[name] != bands[baseline_name]:
                changed_counts[name] = changed_counts[name] + 1

    print("")
    print("Summary: cases where the risk band differs from the current weighting")
    for name in CONFIGS:
        print(name, ":", changed_counts[name], "/", len(cases), "changed")


main()