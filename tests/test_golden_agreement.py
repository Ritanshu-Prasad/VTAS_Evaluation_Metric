"""Golden-set agreement statistics (harness/golden/agreement.py)."""

import csv
import json
import math

import pytest

from harness.golden.agreement import (
    cohen_kappa,
    cluster_bootstrap_kappa,
    gate_threshold,
    krippendorff_alpha_nominal,
    main,
    pair_labels,
)

# ---- Cohen's kappa -----------------------------------------------------------------------


def test_kappa_textbook_example():
    # 50 items: yes/yes 20, yes/no 5, no/yes 10, no/no 15 -> p_o 0.7, p_e 0.5, kappa 0.4
    a = ["y"] * 25 + ["n"] * 25
    b = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
    assert cohen_kappa(a, b) == pytest.approx(0.4)


def test_kappa_perfect_and_degenerate():
    x = ["supported", "contradicted", "unverifiable", "supported"]
    assert cohen_kappa(x, x) == pytest.approx(1.0)
    assert math.isnan(cohen_kappa(["s"] * 5, ["s"] * 5))  # chance agreement is 1
    assert math.isnan(cohen_kappa([], []))
    with pytest.raises(ValueError):
        cohen_kappa(["a"], ["a", "b"])


def test_kappa_matches_sklearn():
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    import numpy as np

    rng = np.random.default_rng(0)
    labels = ["supported", "contradicted", "unverifiable"]
    a = list(rng.choice(labels, 200, p=[0.6, 0.25, 0.15]))
    b = [x if rng.random() < 0.75 else rng.choice(labels) for x in a]
    assert cohen_kappa(a, b) == pytest.approx(sklearn_metrics.cohen_kappa_score(a, b))


# ---- Krippendorff's alpha ----------------------------------------------------------------


def test_alpha_krippendorff_2011_binary_two_coders():
    # Krippendorff (2011), "Computing Krippendorff's alpha-reliability", section A: 0.095
    a = [0, 1, 0, 0, 0, 0, 0, 0, 1, 0]
    b = [1, 1, 1, 0, 0, 1, 0, 0, 0, 0]
    assert krippendorff_alpha_nominal(zip(a, b)) == pytest.approx(0.095, abs=5e-4)


def test_alpha_krippendorff_2011_nominal_with_missing():
    # Same paper, section C: four coders, 12 units, missing values; nominal alpha 0.743
    N = None
    data = [
        [1, 2, 3, 3, 2, 1, 4, 1, 2, N, N, N],
        [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, N, 3],
        [N, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, N],
        [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, N],
    ]
    units = list(zip(*data))
    assert krippendorff_alpha_nominal(units) == pytest.approx(0.743, abs=5e-4)


def test_alpha_perfect_and_undefined():
    x = ["s", "c", "u", "s"]
    assert krippendorff_alpha_nominal(zip(x, x)) == pytest.approx(1.0)
    assert math.isnan(krippendorff_alpha_nominal([("s", "s"), ("s", "s")]))
    assert math.isnan(krippendorff_alpha_nominal([("s", None)]))


# ---- pairing and bootstrap ---------------------------------------------------------------


def test_pair_labels_matches_ids_and_reports_gaps():
    a = [{"claim_id": "c#01", "verdict": "Supported "}, {"claim_id": "c#02", "verdict": ""},
         {"claim_id": "c#03", "verdict": "contradicted"}]
    b = [{"claim_id": "c#03", "verdict": "unverifiable"}, {"claim_id": "c#01", "verdict": "supported"},
         {"claim_id": "c#02", "verdict": "supported"}, {"claim_id": "d#01", "verdict": "supported"}]
    p = pair_labels(a, b)
    assert p["claim_ids"] == ["c#01", "c#03"]
    assert p["a"] == ["supported", "contradicted"] and p["b"] == ["supported", "unverifiable"]
    assert p["only_in_b"] == ["d#01"] and p["unlabelled_a"] == 1


def test_pair_labels_rejects_bad_label_and_duplicates():
    with pytest.raises(ValueError, match="allowed"):
        pair_labels([{"claim_id": "x", "verdict": "maybe"}], [])
    with pytest.raises(ValueError, match="duplicate"):
        pair_labels([{"claim_id": "x", "verdict": "supported"}] * 2, [])


def test_cluster_bootstrap_is_seeded_and_brackets_point():
    import numpy as np

    rng = np.random.default_rng(1)
    labels = ["supported", "contradicted", "unverifiable"]
    a = list(rng.choice(labels, 300))
    b = [x if rng.random() < 0.8 else rng.choice(labels) for x in a]
    clusters = [i // 3 for i in range(300)]
    r1 = cluster_bootstrap_kappa(a, b, clusters, n_resamples=200, seed=3)
    r2 = cluster_bootstrap_kappa(a, b, clusters, n_resamples=200, seed=3)
    assert r1 == r2
    assert r1["low"] <= r1["point"] <= r1["high"]


def test_gate_threshold_read_from_gates_yaml():
    pytest.importorskip("yaml")
    assert gate_threshold() == pytest.approx(0.7)


# ---- end to end on sample files ----------------------------------------------------------

FIELDS = ["claim_id", "caption_id", "image_id", "source", "block", "type", "claim_text",
          "verdict", "visible_count", "note"]


def _write_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})


def _sample_rows():
    base = []
    verdicts = ["supported", "supported", "contradicted", "unverifiable", "supported", "contradicted"]
    for i, v in enumerate(verdicts * 5):
        src = ["blip", "llava_short", "human"][i % 3]
        img = 100 + i // 3
        base.append({"claim_id": f"{src}-{img}#{i:02d}", "caption_id": f"{src}-{img}",
                     "image_id": str(img), "source": src, "block": "P" if i < 15 else "U",
                     "type": ["object", "count", "attribute", "relation", "scene"][i % 5],
                     "claim_text": "x", "verdict": v})
    return base


def test_main_on_sample_csv_files(tmp_path, capsys):
    a = _sample_rows()
    b = [dict(r) for r in a]
    for k in (2, 9, 20):           # three disagreements
        b[k]["verdict"] = "supported" if a[k]["verdict"] != "supported" else "unverifiable"
    _write_csv(tmp_path / "A.csv", a)
    _write_csv(tmp_path / "B.csv", b)
    out = tmp_path / "summary.json"
    res = main([str(tmp_path / "A.csv"), str(tmp_path / "B.csv"), "--out", str(out),
                "--resamples", "200"])
    assert res["n"] == 30
    assert res["percent_agreement"] == pytest.approx(27 / 30)
    assert res["cohen_kappa"] == pytest.approx(
        cohen_kappa([r["verdict"] for r in a], [r["verdict"] for r in b]))
    assert set(res["by_source"]) == {"blip", "llava_short", "human"}
    assert set(res["by_type"]) == {"object", "count", "attribute", "relation", "scene"}
    assert sum(map(sum, res["confusion_rows_A_cols_B"])) == 30
    assert json.loads(out.read_text(encoding="utf-8"))["n"] == 30
    assert "kappa=" in capsys.readouterr().out

    only_p = main([str(tmp_path / "A.csv"), str(tmp_path / "B.csv"), "--blocks", "P",
                   "--resamples", "50"])
    assert only_p["n"] == 15


def test_main_reads_xlsx(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    rows = _sample_rows()
    for name in ("A", "B"):
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "claims"
        ws.append(FIELDS + ["claude_prelabel_verdict"])
        for r in rows:
            ws.append([r.get(k, "") for k in FIELDS] + ["contradicted"])
        wb.save(tmp_path / f"{name}.xlsx")
    res = main([str(tmp_path / "A.xlsx"), str(tmp_path / "B.xlsx"), "--resamples", "50"])
    assert res["n"] == 30 and res["cohen_kappa"] == pytest.approx(1.0)
    if res.get("gate_F_kappa", {}).get("threshold") is not None:
        assert res["gate_F_kappa"]["pass"] is True
