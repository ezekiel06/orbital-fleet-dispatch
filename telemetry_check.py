"""Step 8: A problem detector trained on REAL satellite telemetry.

Data: ESA's OPS-SAT anomaly dataset (OPSSAT-AD, CC-BY 4.0, zenodo.org/records/12588359).
2,123 short telemetry segments from 9 real channels (3 magnetometer, 6 sun-sensor
photodiodes). ESA engineers labeled each segment normal or anomalous. The dataset
comes with a fixed train/test split, so results are comparable with published ones.

Each segment is summarized by 18 ready-made features (average, spread, number of
spikes, how jumpy it is, gaps in the data, ...). Three detectors are compared:

  RULE        simple rule: flag a segment if its spread or jumpiness is far above
              what's normal for that channel (the kind of fixed alarm limit most
              small-satellite teams use)
  UNSUPERVISED  Isolation Forest, trained only on NORMAL segments: it learns what
              normal looks like and flags anything unusual. Works without labels.
  SUPERVISED  Random Forest, trained on labeled normal + anomalous segments.
              Best when you have labeled history.

Scores (all on the held-out test segments):
  precision = of the alarms raised, how many were real problems
  recall    = of the real problems, how many were caught
  F1        = one number balancing the two (1.0 = perfect)

Usage:  python telemetry_check.py
Saves:  telemetry_results.json
"""
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef

FEATURES = ["duration", "len", "mean", "var", "std", "kurtosis", "skew", "n_peaks",
            "smooth10_n_peaks", "smooth20_n_peaks", "diff_peaks", "diff2_peaks", "diff_var",
            "diff2_var", "gaps_squared", "len_weighted", "var_div_duration", "var_div_len"]
CHANNELS = {"CADC0872": "magnetometer X", "CADC0873": "magnetometer Y", "CADC0874": "magnetometer Z",
            "CADC0884": "sun sensor 1", "CADC0886": "sun sensor 2", "CADC0888": "sun sensor 3",
            "CADC0890": "sun sensor 4", "CADC0892": "sun sensor 5", "CADC0894": "sun sensor 6"}


def scores(y, p):
    return {"precision": round(precision_score(y, p, zero_division=0), 3),
            "recall": round(recall_score(y, p), 3),
            "f1": round(f1_score(y, p), 3),
            "mcc": round(matthews_corrcoef(y, p), 3),
            "alarms": int(p.sum()), "caught": int((p & y).sum()), "missed": int((~p & y).sum()),
            "false_alarms": int((p & ~y).sum())}


def main():
    d = pd.read_csv("opssat/dataset.csv")
    tr, te = d[d.train == 1], d[d.train == 0]
    y_te = te.anomaly.values.astype(bool)
    X_tr = np.log1p(np.abs(tr[FEATURES].values)) * np.sign(tr[FEATURES].values)
    X_te = np.log1p(np.abs(te[FEATURES].values)) * np.sign(te[FEATURES].values)
    print(f"Real OPS-SAT telemetry: {len(tr)} training segments, {len(te)} test segments "
          f"({y_te.sum()} real problems in test)\n")

    # RULE: per-channel alarm limits from NORMAL training data (99th percentile)
    nom = tr[tr.anomaly == 0]
    lim = nom.groupby("channel")[["var_div_len", "diff2_var", "gaps_squared"]].quantile(0.99)
    def rule(row):
        L = lim.loc[row.channel]
        return (row.var_div_len > L.var_div_len) or (row.diff2_var > L.diff2_var) or (row.gaps_squared > L.gaps_squared)
    p_rule = te.apply(rule, axis=1).values.astype(bool)

    # UNSUPERVISED: Isolation Forest on normal segments only
    iso = IsolationForest(n_estimators=300, contamination=0.05, random_state=0)
    iso.fit(X_tr[tr.anomaly.values == 0])
    p_iso = iso.predict(X_te) == -1

    # SUPERVISED: Random Forest on labeled training data
    rf = RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=0, n_jobs=-1)
    rf.fit(X_tr, tr.anomaly.values)
    p_rf = rf.predict(X_te).astype(bool)

    res = {"rule": scores(y_te, p_rule), "unsupervised": scores(y_te, p_iso), "supervised": scores(y_te, p_rf)}
    print(f"{'':14}{'RULE':>10}{'UNSUPERV.':>11}{'SUPERV.':>10}")
    for k in ["precision", "recall", "f1", "mcc", "caught", "missed", "false_alarms"]:
        print(f"{k:14}" + "".join(f"{str(res[m][k]):>{w}}" for m, w in [("rule", 10), ("unsupervised", 11), ("supervised", 10)]))

    # which channels the supervised model gets wrong
    te = te.assign(pred=p_rf)
    per = (te.assign(err=lambda x: x.pred != x.anomaly.astype(bool))
             .groupby("channel").agg(segments=("err", "size"), errors=("err", "sum")))
    per["name"] = per.index.map(CHANNELS)
    print("\nSupervised mistakes by channel:")
    for ch, r in per.iterrows():
        print(f"  {r['name']:<15} {int(r.errors):>3} wrong of {int(r.segments)}")
    top = sorted(zip(FEATURES, rf.feature_importances_), key=lambda x: -x[1])[:5]
    print("\nMost useful signals:", ", ".join(f"{f} ({v:.2f})" for f, v in top))

    with open("telemetry_results.json", "w") as f:
        json.dump({"dataset": "ESA OPS-SAT (OPSSAT-AD), CC-BY 4.0", "test_segments": int(len(te)),
                   "test_anomalies": int(y_te.sum()), "results": res,
                   "top_features": [[f, round(float(v), 3)] for f, v in top]}, f, indent=1)
    print("\nSaved telemetry_results.json")


if __name__ == "__main__":
    main()
