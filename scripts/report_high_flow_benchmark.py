"""Build the internal high-flow benchmark from saved prediction artifacts only.

The benchmark asks one narrow question: on the canonical 2024 six-hour high-flow
rows shared by the retained artifacts, how do the saved candidate families compare
after threshold choice is controlled?  It never fits a model, downloads data or
writes outside ``experiments/high-flow-benchmark-20261018``.

Two operating points are reported for every admitted candidate:

``own_threshold``
    The candidate's frozen threshold from its own experiment record.
``fixed_budget``
    A single pooled 1.5-false-alerts-per-100 threshold, selected from the 2024
    benchmark labels separately for each candidate.  This is a retrospective
    matched-budget diagnostic, not a deployable threshold policy.

The benchmark is internal development evidence.  It is not an external benchmark
and it is not a state-of-the-art claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

# Importing the repository metric module must not write bytecode into existing
# experiment or script directories.
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SIGNALS = ROOT / "experiments" / "high-flow-signals-20261018"
RESEARCH = ROOT / "experiments" / "high-flow-20261018"
OUT_DIR = ROOT / "experiments" / "high-flow-benchmark-20261018"

PARTITIONS = ("temporal_validation", "geographic_validation")
PARTITION_LABEL = {"temporal_validation": "temporal", "geographic_validation": "geographic"}
POOLED = "pooled"
FIXED_BUDGET = 1.5
DRAWS = 500
SEED = 42
KEY = ["station", "decision_time"]
ROW_COLUMNS = ["station", "huc", "decision_time", "horizon", "target", "event_time"]
REFERENCE_ID = "signal/c7/linear_conditional"
FROZEN_ID = "signal/c11/winner_forecast"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def frame_digest(frame: pd.DataFrame) -> str:
    ordered = frame.sort_values(KEY + ["horizon"]).reset_index(drop=True)
    return hashlib.sha256(pd.util.hash_pandas_object(ordered, index=False).values.tobytes()).hexdigest()


def json_value(value):
    """Convert numpy and Path values into deterministic JSON values."""
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else round(value, 12)
    if isinstance(value, Path):
        return str(value.relative_to(ROOT))
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def canonical_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["station"] = frame.station.astype(str)
    frame["huc"] = frame.huc.astype(str)
    frame["decision_time"] = pd.to_datetime(frame.decision_time, format="ISO8601", utc=True, errors="raise")
    frame["horizon"] = frame.horizon.astype(int)
    frame["target"] = frame.target.astype(int)
    frame["event_time"] = frame.event_time.fillna("").astype(str)
    frame = frame[ROW_COLUMNS].sort_values(KEY + ["horizon"]).reset_index(drop=True)
    if frame.duplicated(KEY + ["horizon"]).any():
        raise ValueError("Canonical frame has duplicate station/decision_time/horizon keys")
    return frame


def load_reference_frames() -> dict[str, pd.DataFrame]:
    frames = {}
    for part in PARTITIONS:
        path = SIGNALS / "cycle-11" / f"winner_forecast-{part}.csv.gz"
        frame = pd.read_csv(path, dtype={"station": str, "huc": str}, keep_default_na=False)
        frames[part] = canonical_frame(frame)
    return frames


def read_candidate_frame(path: Path, reference: pd.DataFrame, label: str) -> pd.DataFrame:
    """Read one saved prediction file and align it to the canonical rows.

    Rows outside the canonical cohort are dropped, including the historical
    cycle-2 temporal extension.  Every retained label must match the canonical
    frame exactly; a mismatch fails the benchmark instead of being repaired.
    """
    if not path.exists():
        raise FileNotFoundError(path)
    pred = pd.read_csv(path, dtype={"station": str, "huc": str}, keep_default_na=False)
    required = set(ROW_COLUMNS + ["probability"])
    if not required.issubset(pred.columns):
        raise ValueError(f"Missing columns in {path}: {sorted(required - set(pred.columns))}")
    pred = pred[ROW_COLUMNS + ["probability"]].copy()
    pred["station"] = pred.station.astype(str)
    pred["huc"] = pred.huc.astype(str)
    pred["decision_time"] = pd.to_datetime(pred.decision_time, format="ISO8601", utc=True, errors="raise")
    pred["horizon"] = pred.horizon.astype(int)
    pred["target"] = pred.target.astype(int)
    pred["event_time"] = pred.event_time.fillna("").astype(str)
    pred["probability"] = pd.to_numeric(pred.probability, errors="coerce")
    if pred.duplicated(KEY + ["horizon"]).any():
        raise ValueError(f"Duplicate prediction rows in {path}")
    if not np.isfinite(pred.probability.to_numpy(float)).all():
        raise ValueError(f"Non-finite probabilities in {path}")
    aligned = reference.merge(
        pred[["station", "decision_time", "huc", "horizon", "target", "event_time", "probability"]],
        on=["station", "decision_time"],
        how="left",
        suffixes=("", "_prediction"),
        validate="one_to_one",
    )
    if len(aligned) != len(reference) or aligned.probability.isna().any():
        missing = int(aligned.probability.isna().sum())
        raise ValueError(f"{label} does not cover the canonical rows ({missing} missing)")
    for column in ("huc", "horizon", "target", "event_time"):
        if not aligned[column].eq(aligned[column + "_prediction"]).all():
            raise ValueError(f"{label} labels differ on {column}")
    aligned = aligned[ROW_COLUMNS + ["probability"]].copy()
    aligned["probability"] = aligned.probability.astype(float)
    return aligned


def signal_catalog() -> list[dict]:
    candidates = []
    for folder in sorted(SIGNALS.glob("cycle-*"), key=lambda value: int(value.name.split("-")[1])):
        cycle = int(folder.name.split("-")[1])
        records = json.loads((folder / "results.json").read_text())
        for record in records:
            name = record["name"]
            paths = {part: folder / f"{name}-{part}.csv.gz" for part in PARTITIONS}
            for path in paths.values():
                if not path.exists():
                    raise FileNotFoundError(path)
            candidates.append(dict(
                id=f"signal/c{cycle}/{name}",
                lineage="high-flow-signals-20261018",
                cycle=cycle,
                name=name,
                policy="frozen",
                threshold=float(record["threshold"]),
                threshold_source=str((folder / "results.json").relative_to(ROOT)),
                prediction_paths={part: str(path.relative_to(ROOT)) for part, path in paths.items()},
                record_metrics=record.get("metrics"),
                record_comparable=True,
            ))
    return candidates


def research_catalog() -> list[dict]:
    candidates = []
    for cycle in (2, 3, 4):
        payload = json.loads((RESEARCH / f"cycle-{cycle}.json").read_text())
        for record in payload["records"]:
            name = record["name"]
            paths = {part: RESEARCH / "predictions" / f"cycle-{cycle}-{name}-{part}.csv.gz" for part in PARTITIONS}
            for path in paths.values():
                if not path.exists():
                    raise FileNotFoundError(path)
            candidates.append(dict(
                id=f"research/c{cycle}/{name}/{record['policy']}",
                lineage="high-flow-20261018",
                cycle=cycle,
                name=name,
                policy=record["policy"],
                threshold=float(record["threshold"]),
                threshold_source=str((RESEARCH / f"cycle-{cycle}.json").relative_to(ROOT)),
                prediction_paths={part: str(path.relative_to(ROOT)) for part, path in paths.items()},
                record_metrics=record.get("metrics"),
                # Cycle 2's frozen metrics were computed on the historical
                # 75,527-row temporal extension, not on the canonical subset.
                record_comparable=cycle != 2,
            ))
    return candidates


def pooled_budget_threshold(frame: pd.DataFrame, probabilities: np.ndarray, budget: float) -> float:
    """Lowest inclusive threshold whose pooled false alerts stay within budget.

    This is the same tie-preserving rule used by the repository's research
    protocol, applied to the pooled 2024 benchmark rows.  It deliberately uses
    benchmark labels because the purpose is an outcome-matched diagnostic; the
    benchmark labels it as such.
    """
    y = frame.target.to_numpy(int)
    p = np.asarray(probabilities, float)
    if len(y) != len(p) or not np.isfinite(p).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("Invalid threshold inputs")
    order = np.argsort(-p, kind="stable")
    scores = p[order]
    stops = np.flatnonzero(np.r_[scores[1:] != scores[:-1], True])
    false_alerts = np.cumsum(1 - y[order])[stops]
    feasible = 100 * false_alerts / len(y) <= budget + 1e-9
    if not feasible.any():
        return float(np.nextafter(p.max(), np.inf))
    return float(scores[stops][np.flatnonzero(feasible)[-1]])


class ClusterBootstrap:
    """Paired HUC4-cluster bootstrap with vectorized AP, recall and alert counts."""

    def __init__(self, frame: pd.DataFrame, draws: int = DRAWS, seed: int = SEED):
        self.frame = frame.reset_index(drop=True)
        self.y = self.frame.target.to_numpy(int)
        codes, groups = pd.factorize(self.frame.huc.str[:4])
        self.codes = codes
        self.groups = groups
        self.clusters = len(groups)
        self.rows = np.bincount(codes, minlength=self.clusters)
        event_index = self.frame.loc[self.y == 1].drop_duplicates(["station", "event_time"]).index.to_numpy()
        self.events = np.bincount(codes[event_index], minlength=self.clusters)
        self.positive_rows = np.bincount(codes[self.y == 1], minlength=self.clusters)
        rng = np.random.default_rng(seed)
        self.weights = np.array([
            np.bincount(rng.integers(self.clusters, size=self.clusters), minlength=self.clusters)
            for _ in range(draws)
        ], dtype=float)
        self.one = np.ones((1, self.clusters), dtype=float)

    def accumulators(self, probabilities: np.ndarray) -> dict:
        p = np.asarray(probabilities, float)
        order = np.argsort(-p, kind="stable")
        scores = p[order]
        y = self.y[order]
        codes = self.codes[order]
        positive = np.flatnonzero(y == 1)
        if not len(positive):
            raise ValueError("Cannot compute AP without positive rows")
        last = np.searchsorted(-scores, -scores[positive], side="right") - 1
        onehot_rows = np.zeros((len(scores), self.clusters), dtype=np.int64)
        onehot_rows[np.arange(len(scores)), codes] = 1
        onehot_positive = np.zeros_like(onehot_rows)
        onehot_positive[np.arange(len(scores)), codes] = y
        row_cumulative = np.cumsum(onehot_rows, axis=0)
        positive_cumulative = np.cumsum(onehot_positive, axis=0)
        return dict(
            row_counts=row_cumulative[last],
            positive_counts=positive_cumulative[last],
            positive_clusters=codes[positive],
        )

    def weighted_ap(self, accumulators: dict, weights: np.ndarray) -> np.ndarray:
        true_positive = weights @ accumulators["positive_counts"].T
        all_rows = weights @ accumulators["row_counts"].T
        row_weight = weights[:, accumulators["positive_clusters"]]
        numerator = (row_weight * (true_positive / np.maximum(all_rows, 1))).sum(axis=1)
        return numerator / np.maximum(weights @ self.positive_rows, 1e-12)

    def threshold_counts(self, probabilities: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
        p = np.asarray(probabilities, float)
        alert = p >= threshold
        hit_index = np.flatnonzero((self.y == 1) & alert)
        false_index = np.flatnonzero((self.y == 0) & alert)
        return (np.bincount(self.codes[hit_index], minlength=self.clusters),
                np.bincount(self.codes[false_index], minlength=self.clusters))

    def event_recall_values(self, hit_difference: np.ndarray, weights: np.ndarray) -> np.ndarray:
        return (weights @ hit_difference) / np.maximum(weights @ self.events, 1e-12)

    def false_alert_values(self, false_difference: np.ndarray, weights: np.ndarray) -> np.ndarray:
        return 100 * (weights @ false_difference) / np.maximum(weights @ self.rows, 1e-12)


def interval(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def paired_outputs(
    bootstrap: ClusterBootstrap,
    candidate_accumulators: dict,
    candidate_threshold: float,
    candidate_probabilities: np.ndarray,
    reference_accumulators: dict,
    reference_threshold: float,
    reference_probabilities: np.ndarray,
) -> dict:
    candidate_hit, candidate_false = bootstrap.threshold_counts(candidate_probabilities, candidate_threshold)
    reference_hit, reference_false = bootstrap.threshold_counts(reference_probabilities, reference_threshold)
    candidate_ap = bootstrap.weighted_ap(candidate_accumulators, bootstrap.weights)
    reference_ap = bootstrap.weighted_ap(reference_accumulators, bootstrap.weights)
    ap_difference = candidate_ap - reference_ap
    recall_difference = bootstrap.event_recall_values(candidate_hit - reference_hit, bootstrap.weights)
    false_difference = bootstrap.false_alert_values(candidate_false - reference_false, bootstrap.weights)
    point_candidate_ap = bootstrap.weighted_ap(candidate_accumulators, bootstrap.one)
    point_reference_ap = bootstrap.weighted_ap(reference_accumulators, bootstrap.one)
    point_ap = float(np.asarray(point_candidate_ap).reshape(-1)[0] - np.asarray(point_reference_ap).reshape(-1)[0])
    point_recall = bootstrap.event_recall_values(candidate_hit - reference_hit, bootstrap.one)
    point_false = bootstrap.false_alert_values(candidate_false - reference_false, bootstrap.one)
    return dict(
        ap_difference=point_ap,
        ap_difference_95=interval(ap_difference),
        event_recall_difference=float(np.asarray(point_recall).reshape(-1)[0]),
        event_recall_difference_95=interval(recall_difference),
        false_alerts_per_100_difference=float(np.asarray(point_false).reshape(-1)[0]),
        false_alerts_per_100_difference_95=interval(false_difference),
        clusters=bootstrap.clusters,
        draws=DRAWS,
        seed=SEED,
        point_candidate_ap=float(np.asarray(point_candidate_ap).reshape(-1)[0]),
        point_reference_ap=float(np.asarray(point_reference_ap).reshape(-1)[0]),
        valid_ap_draws=int(np.isfinite(candidate_ap - reference_ap).sum()),
    )


def _event_leads(frame: pd.DataFrame, alert: np.ndarray) -> pd.Series:
    """Earliest alert per unique (station, event_time), measured in hours."""
    hits = frame.loc[frame.target.eq(1) & alert, ["station", "event_time", "decision_time"]].copy()
    if hits.empty:
        return pd.Series(dtype=float)
    hits["lead"] = (pd.to_datetime(hits.event_time, utc=True) - hits.decision_time).dt.total_seconds() / 3600
    return hits.groupby(["station", "event_time"], sort=False).lead.max()


def _summary_metrics(frame: pd.DataFrame, probabilities: np.ndarray, threshold: float) -> dict:
    """Repository metric definitions, applied directly to canonical frames."""
    y = frame.target.to_numpy(int)
    p = np.asarray(probabilities, float)
    alert = p >= threshold
    true_positive = int(np.sum(alert & (y == 1)))
    false_positive = int(np.sum(alert & (y == 0)))
    alerts = int(alert.sum())
    positives = int(y.sum())
    leads = _event_leads(frame, alert)
    event_count = int(frame.loc[y == 1].drop_duplicates(["station", "event_time"]).shape[0])
    false_times = frame.loc[alert & (y == 0), ["station", "decision_time"]].copy()
    gaps = false_times.sort_values(["station", "decision_time"]).groupby("station", sort=False).decision_time.diff()
    return dict(
        rows=len(frame),
        positives=positives,
        events=event_count,
        detected_events=int(len(leads)),
        event_recall=len(leads) / event_count if event_count else None,
        ap=float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else None,
        brier=float(brier_score_loss(y, p)),
        precision=true_positive / alerts if alerts else None,
        row_recall=true_positive / positives if positives else None,
        false_alerts_per_100=100 * false_positive / len(frame),
        false_warning_episodes=int((gaps.isna() | gaps.gt(pd.Timedelta(hours=24))).sum()),
        median_lead_hours=float(leads.median()) if len(leads) else None,
    )


def compact_metrics(frame: pd.DataFrame, probabilities: np.ndarray, threshold: float) -> dict:
    measured = _summary_metrics(frame, probabilities, threshold)
    regions = {}
    labels = frame.huc.str[:2]
    for key, group in frame.groupby(labels, sort=True):
        positions = group.index.to_numpy()
        regions[str(key)] = _summary_metrics(group.reset_index(drop=True), np.asarray(probabilities)[positions], threshold)
    worst = max(regions, key=lambda key: regions[key]["false_alerts_per_100"]) if regions else None
    return dict(
        **{key: measured[key] for key in
           ("rows", "positives", "events", "detected_events", "event_recall", "ap", "brier",
            "precision", "row_recall", "false_alerts_per_100", "false_warning_episodes", "median_lead_hours")},
        worst_huc2=worst,
        worst_huc2_false_alerts_per_100=None if worst is None else regions[worst]["false_alerts_per_100"],
        huc2={key: dict(rows=value["rows"], events=value["events"], ap=value["ap"],
                        false_alerts_per_100=value["false_alerts_per_100"])
              for key, value in regions.items()},
    )


def compact_lead_delta(frame: pd.DataFrame, control_probabilities: np.ndarray, candidate_probabilities: np.ndarray,
                       control_threshold: float, candidate_threshold: float) -> dict:
    control = _event_leads(frame, np.asarray(control_probabilities) >= control_threshold)
    candidate = _event_leads(frame, np.asarray(candidate_probabilities) >= candidate_threshold)
    joint = pd.concat([control.rename("control"), candidate.rename("candidate")], axis=1, join="inner")
    control_median = float(joint.control.median()) if len(joint) else None
    candidate_median = float(joint.candidate.median()) if len(joint) else None
    return dict(jointly_detected_events=int(len(joint)), control_median_lead_hours=control_median,
                candidate_median_lead_hours=candidate_median,
                median_lead_delta_hours=None if not len(joint) else candidate_median - control_median)


def build_group(
    frames: dict[str, pd.DataFrame],
    probabilities: dict[str, np.ndarray],
    own_threshold: float,
    reference: dict,
    bootstraps: dict[str, ClusterBootstrap],
    accumulators: dict[str, dict],
    reference_accumulators: dict[str, dict],
    pooled_frame: pd.DataFrame,
    pooled_probabilities: np.ndarray,
) -> dict:
    fixed_threshold = pooled_budget_threshold(pooled_frame, pooled_probabilities, FIXED_BUDGET)
    result = dict(thresholds=dict(own=own_threshold, fixed_budget=fixed_threshold,
                                  fixed_budget_target=FIXED_BUDGET), partitions={})
    for part in (*PARTITIONS, POOLED):
        frame = pooled_frame if part == POOLED else frames[part]
        p = pooled_probabilities if part == POOLED else probabilities[part]
        bootstrap = bootstraps[part]
        reference_p = reference["pooled_probabilities"] if part == POOLED else reference["probabilities"][part]
        own_metrics = compact_metrics(frame, p, own_threshold)
        fixed_metrics = compact_metrics(frame, p, fixed_threshold)
        current_accumulators = accumulators[part]
        paired_own = paired_outputs(bootstrap, current_accumulators, own_threshold, p,
                                    reference_accumulators[part], reference["own_threshold"], reference_p)
        paired_fixed = paired_outputs(bootstrap, current_accumulators, fixed_threshold, p,
                                      reference_accumulators[part], reference["fixed_threshold"], reference_p)
        lead_own = compact_lead_delta(frame, reference_p, p,
                                      reference["own_threshold"], own_threshold)
        lead_fixed = compact_lead_delta(frame, reference_p, p,
                                        reference["fixed_threshold"], fixed_threshold)
        result["partitions"][part] = dict(
            own=dict(metrics=own_metrics, paired_vs_reference=paired_own, lead_vs_reference=lead_own),
            fixed_budget=dict(metrics=fixed_metrics, paired_vs_reference=paired_fixed,
                              lead_vs_reference=lead_fixed),
        )
    return result


def ranking(groups: list[dict], partition: str, operating_point: str, key: str, reverse: bool = True) -> list[dict]:
    rows = []
    for group in groups:
        values = group["partitions"][partition][operating_point]["metrics"]
        rows.append(dict(group_id=group["group_id"], representative=group["representative_id"], value=values[key]))
    rows.sort(key=lambda row: (row["value"] is None, -(row["value"] or 0) if reverse else (row["value"] or 0)))
    return rows


def frozen_comparison(groups: list[dict]) -> list[dict]:
    frozen = next((group for group in groups if FROZEN_ID in group["member_ids"]), None)
    if frozen is None:
        raise ValueError("Frozen candidate not found in benchmark groups")
    output = []
    for point in ("own", "fixed_budget"):
        base = frozen["partitions"]
        for group in groups:
            candidate = group["partitions"]
            t_ap = candidate["temporal_validation"][point]["metrics"]["ap"]
            t_events = candidate["temporal_validation"][point]["metrics"]["detected_events"]
            g_ap = candidate["geographic_validation"][point]["metrics"]["ap"]
            g_events = candidate["geographic_validation"][point]["metrics"]["detected_events"]
            ft_ap = base["temporal_validation"][point]["metrics"]["ap"]
            ft_events = base["temporal_validation"][point]["metrics"]["detected_events"]
            fg_ap = base["geographic_validation"][point]["metrics"]["ap"]
            fg_events = base["geographic_validation"][point]["metrics"]["detected_events"]
            values = {
                "temporal_ap_delta": None if t_ap is None or ft_ap is None else t_ap - ft_ap,
                "temporal_event_delta": t_events - ft_events,
                "geographic_ap_delta": None if g_ap is None or fg_ap is None else g_ap - fg_ap,
                "geographic_event_delta": g_events - fg_events,
                "no_worse_on_all_four": bool(t_ap is not None and ft_ap is not None and g_ap is not None and fg_ap is not None
                                             and t_ap >= ft_ap and g_ap >= fg_ap and t_events >= ft_events and g_events >= fg_events),
                "better_on_at_least_one": bool(t_ap is not None and ft_ap is not None and g_ap is not None and fg_ap is not None
                                                and (t_ap > ft_ap or g_ap > fg_ap or t_events > ft_events or g_events > fg_events)),
            }
            values["dominates_frozen"] = bool(values["no_worse_on_all_four"] and values["better_on_at_least_one"])
            output.append(dict(operating_point=point, group_id=group["group_id"],
                               representative=group["representative_id"],
                               is_frozen=FROZEN_ID in group["member_ids"], **values))
    return output


def excluded_families() -> list[dict]:
    """Catalog the declared earlier families and why they are not on the cohort."""
    return [
        dict(family="experiments/sprint-20261017-hydrology", comparable=False,
             reason="Daily 1-day and 2-day targets on forward/geographic daily cohorts; no shared six-hour 2024 cohort.",
             observed=dict(prediction_example="baseline_persistence_h1.forward_test.csv.gz", rows=24915,
                           positives=240, horizon="daily 1-day and 2-day targets"),
             evidence=["experiments/sprint-20261017-hydrology/protocol.json",
                       "experiments/sprint-20261017-hydrology/predictions/*.csv.gz"]),
        dict(family="experiments/sprint-20261017-shift-hydrology", comparable=False,
             reason="Six-, twelve- and twenty-four-hour 2025 final-period populations; event labels and gauges differ from the canonical 2024 rows.",
             observed=dict(test_rows=69411, horizons=[6, 12, 24], positives=564, period="2025"),
             evidence=["experiments/sprint-20261017-shift-hydrology/protocol.json",
                       "experiments/sprint-20261017-shift-hydrology/test.csv.gz"]),
        dict(family="experiments/sprint-20261017-mlp-onset", comparable=False,
             reason="Six-, twelve- and twenty-four-hour 2025 development/geographic populations; not the canonical 2024 cohort.",
             observed=dict(test_rows=79233, horizons=[6, 12, 24], positives=939, period="2025"),
             evidence=["experiments/sprint-20261017-mlp-onset/protocol.json",
                       "experiments/sprint-20261017-mlp-onset/test.csv.gz"]),
        dict(family="experiments/sprint-20261017-mlp-hybrid", comparable=False,
             reason="Six-, twelve- and twenty-four-hour 2025 temporal/geographic test populations; different gauges and inspected outcomes.",
             observed=dict(temporal_rows_horizon6=10340, geographic_rows_horizon6=7848,
                           horizons=[6, 12, 24], period="2025"),
             evidence=["experiments/sprint-20261017-mlp-hybrid/protocol.json",
                       "experiments/sprint-20261017-mlp-hybrid/predictions/*.csv.gz"]),
        dict(family="experiments/sprint-20261017-mlp-geo", comparable=False,
             reason="Six-, twelve- and twenty-four-hour 2025 development/geographic populations; different gauges and inspected outcomes.",
             observed=dict(test_rows=70992, horizons=[6, 12, 24], positives=913, period="2025"),
             evidence=["experiments/sprint-20261017-mlp-geo/protocol.json",
                       "experiments/sprint-20261017-mlp-geo/test.csv.gz"]),
        dict(family="experiments/sprint-20261017-transformer", comparable=False,
             reason="Archived forecast maximum-gust target and 2026 station-date cohorts; different target and observation contract.",
             observed=dict(prediction_rows=36800, models=5, cohorts="inspected and fresh temporal/geographic",
                           period="2026"),
             evidence=["experiments/sprint-20261017-transformer/selection.json",
                       "experiments/sprint-20261017-transformer/evaluation.json"]),
        dict(family="experiments/sprint-20261017-analog", comparable=False,
             reason="Daily 2023 fresh-gauge test population with its own event labels and one-day horizon.",
             observed=dict(test_rows=12389, horizon=1, positives=294, period="2023"),
             evidence=["experiments/sprint-20261017-analog/protocol.json",
                       "experiments/sprint-20261017-analog/test.csv.gz"]),
        dict(family="experiments/high-flow-20261018/cycle-1", comparable=False,
             reason="Historical cycle-1 predictions cover only 14,157 temporal rows and 10,219 geographic rows; the full canonical cohort is unavailable.",
             observed=dict(temporal_rows=14157, geographic_rows=10219),
             evidence=["experiments/high-flow-20261018/cycle-1.json",
                       "experiments/high-flow-20261018/predictions/cycle-1-*-validation.csv.gz"]),
        dict(family="experiments/high-flow-signals-20261018/upstream", comparable=False,
             reason="Separate downstream/upstream paired-gauge population; 14 events and 3,811 paired rows in the prior report.",
             observed=dict(development_rows=13473, prior_paired_rows=3811, prior_events=14),
             evidence=["docs/HIGH_FLOW_SIGNAL_RESULTS.md",
                       "experiments/high-flow-signals-20261018/upstream/development.csv.gz"]),
    ]


def source_metric_reproduction(candidates: list[dict], candidate_index: list[dict], groups: list[dict]) -> dict:
    """Compare benchmark-side scoring with each artifact's own frozen record."""
    by_id = {candidate["id"]: candidate for candidate in candidates}
    group_by_id = {group["group_id"]: group for group in groups}
    compared = 0
    maximum = 0.0
    mismatches = []
    for item in candidate_index:
        candidate = by_id[item["id"]]
        if not candidate.get("record_comparable") or not candidate.get("record_metrics"):
            continue
        group_id = hashlib.sha256(item["group_key"].encode()).hexdigest()[:16]
        group = group_by_id[group_id]
        for part in PARTITIONS:
            record = candidate["record_metrics"].get(part)
            if not isinstance(record, dict):
                continue
            measured = group["partitions"][part]["own"]["metrics"]
            for key in ("detected_events", "ap", "brier", "false_alerts_per_100", "median_lead_hours"):
                expected, actual = record.get(key), measured.get(key)
                if expected is None or actual is None:
                    continue
                difference = abs(float(expected) - float(actual))
                compared += 1
                maximum = max(maximum, difference)
                if difference > 1e-6:
                    mismatches.append(dict(id=item["id"], partition=part, metric=key,
                                           record=expected, benchmark=actual,
                                           difference=difference))
    mismatches.sort(key=lambda item: item["difference"], reverse=True)
    return dict(compared_values=compared, max_abs_difference=maximum,
                mismatch_count=len(mismatches), mismatches=mismatches[:50])


def build_benchmark(out_dir: Path) -> dict:
    reference_frames = load_reference_frames()
    cohort = {
        part: dict(
            rows=len(frame),
            stations=int(frame.station.nunique()),
            huc4=int(frame.huc.str[:4].nunique()),
            positives=int(frame.target.sum()),
            events=int(frame.loc[frame.target.eq(1)].drop_duplicates(["station", "event_time"]).shape[0]),
            sha256=frame_digest(frame),
            source=str((SIGNALS / "cycle-11" / f"winner_forecast-{part}.csv.gz").relative_to(ROOT)),
        )
        for part, frame in reference_frames.items()
    }
    candidates = signal_catalog() + research_catalog()
    print("candidates", len(candidates), "reading canonical rows", flush=True)

    # Read each unique prediction pair once.  The cache key keeps a candidate's
    # frozen threshold separate from a different threshold on the same scores.
    cache: dict[tuple, dict] = {}
    signature_cache: dict[tuple, dict] = {}
    groups: list[dict] = []
    member_ids: dict[tuple, list[str]] = {}
    candidate_index = []
    reference_bundle = None
    for candidate in candidates:
        frames = {part: read_candidate_frame(Path(candidate["prediction_paths"][part]), reference_frames[part], candidate["id"])
                  for part in PARTITIONS}
        probabilities = {part: frames[part].probability.to_numpy(float) for part in PARTITIONS}
        signature = tuple(
            hashlib.sha256(np.asarray(probabilities[part], dtype="<f8").tobytes()).hexdigest()
            for part in PARTITIONS
        )
        group_key = signature + (round(candidate["threshold"], 12),)
        candidate_index.append(dict(id=candidate["id"], group_key=repr(group_key), source=candidate["lineage"],
                                    cycle=candidate["cycle"], name=candidate["name"], policy=candidate["policy"],
                                    threshold=candidate["threshold"], threshold_source=candidate["threshold_source"],
                                    prediction_paths=candidate["prediction_paths"]))
        cache.setdefault(group_key, dict(signature=signature, own_threshold=candidate["threshold"]))
        if signature not in signature_cache:
            pooled = pd.concat([frames[part] for part in PARTITIONS], ignore_index=True)
            signature_cache[signature] = dict(
                frames=frames,
                probabilities=probabilities,
                pooled_frame=pooled,
                pooled_probabilities=np.concatenate([probabilities[part] for part in PARTITIONS]),
            )
        member_ids.setdefault(group_key, []).append(candidate["id"])
        if candidate["id"] == REFERENCE_ID:
            reference_bundle = signature_cache[signature]

    if reference_bundle is None:
        # The reference may already exist under a different frozen threshold if
        # the source artifact changed.  Prefer the exact declared candidate.
        raise ValueError(f"Reference candidate {REFERENCE_ID} was not found")
    reference_own_threshold = next(candidate["threshold"] for candidate in candidates
                                   if candidate["id"] == REFERENCE_ID)
    reference = dict(
        frames=reference_bundle["frames"],
        probabilities=reference_bundle["probabilities"],
        pooled_probabilities=reference_bundle["pooled_probabilities"],
        own_threshold=reference_own_threshold,
        fixed_threshold=None,
    )
    reference_pooled_frame = pd.concat([reference_frames[part] for part in PARTITIONS], ignore_index=True)
    reference["fixed_threshold"] = pooled_budget_threshold(
        reference_pooled_frame, reference["pooled_probabilities"], FIXED_BUDGET)

    bootstraps = {part: ClusterBootstrap(reference_frames[part]) for part in PARTITIONS}
    bootstraps[POOLED] = ClusterBootstrap(reference_pooled_frame)
    reference_accumulators = {
        part: bootstraps[part].accumulators(reference["pooled_probabilities"] if part == POOLED
                                            else reference["probabilities"][part])
        for part in (*PARTITIONS, POOLED)
    }
    accumulator_cache: dict[tuple, dict] = {}

    print("distinct groups", len(cache), "building metrics", flush=True)
    for index, (group_key, item) in enumerate(cache.items(), start=1):
        meta = signature_cache[item["signature"]]
        accumulators = {}
        for part in (*PARTITIONS, POOLED):
            cache_key = (item["signature"], part)
            if cache_key not in accumulator_cache:
                accumulator_cache[cache_key] = bootstraps[part].accumulators(
                    meta["pooled_probabilities"] if part == POOLED else meta["probabilities"][part])
            accumulators[part] = accumulator_cache[cache_key]
        group = build_group(meta["frames"], meta["probabilities"], item["own_threshold"], reference,
                            bootstraps, accumulators, reference_accumulators,
                            meta["pooled_frame"], meta["pooled_probabilities"])
        group_id = hashlib.sha256(repr(group_key).encode()).hexdigest()[:16]
        group["group_id"] = group_id
        group["member_ids"] = sorted(member_ids[group_key])
        group["representative_id"] = sorted(member_ids[group_key])[0]
        groups.append(group)
        if index % 25 == 0:
            print("built groups", index, "of", len(cache), flush=True)

    groups.sort(key=lambda value: value["representative_id"])
    frozen = next(group for group in groups if FROZEN_ID in group["member_ids"])
    ranked = dict(
        own_temporal_ap=ranking(groups, "temporal_validation", "own", "ap"),
        own_geographic_ap=ranking(groups, "geographic_validation", "own", "ap"),
        own_pooled_event_recall=ranking(groups, POOLED, "own", "event_recall"),
        fixed_temporal_event_recall=ranking(groups, "temporal_validation", "fixed_budget", "event_recall"),
        fixed_geographic_event_recall=ranking(groups, "geographic_validation", "fixed_budget", "event_recall"),
        fixed_pooled_ap=ranking(groups, POOLED, "fixed_budget", "ap"),
    )
    result = dict(
        benchmark_id="planit-internal-high-flow-20261018",
        label="Internal development benchmark; no external or SOTA claim.",
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        commands=[
            ".venv/bin/python -B scripts/report_high_flow_benchmark.py",
            ".venv/bin/python -B scripts/report_high_flow_benchmark.py --out-dir experiments/high-flow-benchmark-20261018",
        ],
        scope=dict(
            target="six-hour high-flow onset above the station reference percentile",
            benchmark_rows="2024 temporal and geographic validation partitions from the signal lineage",
            event_definition="unique (station, event_time) positive rows; 24-hour quiet period and two-hour assumed observation delay inherited from the frozen protocol",
            evidence="saved per-row probabilities only; no refitting, downloads, or production changes",
        ),
        cohort=cohort,
        operating_points=dict(
            own_threshold="the threshold recorded by each candidate's own experiment",
            fixed_budget=dict(
                label="fixed_budget_oracle_2024",
                pooled_false_alerts_per_100=FIXED_BUDGET,
                note="one threshold per candidate selected on the pooled 2024 benchmark labels; a retrospective matched-budget diagnostic, not a deployment policy",
            ),
        ),
        reference=dict(
            id=REFERENCE_ID,
            choice="strongest calibrated physical comparator in the retained cycle-7 report",
            own_threshold=reference["own_threshold"],
            fixed_threshold=reference["fixed_threshold"],
            source=str((SIGNALS / "cycle-7" / "results.json").relative_to(ROOT)),
        ),
        comparability=dict(
            admitted=[
                dict(family="experiments/high-flow-signals-20261018/cycle-0..13",
                     candidates=len(signal_catalog()),
                     basis="all saved prediction files resolve to the same canonical temporal and geographic row keys and labels"),
                dict(family="experiments/high-flow-20261018/cycle-2,3,4",
                     candidates=sum(1 for item in candidate_index if item["source"] == "high-flow-20261018"),
                     basis="cycle 2 predictions contain the canonical rows as a subset; cycles 3 and 4 cover them directly"),
            ],
            excluded=excluded_families(),
        ),
        candidate_index=candidate_index,
        candidate_groups=groups,
        frozen_candidate=dict(id=FROZEN_ID, group_id=frozen["group_id"], metrics=frozen["partitions"]),
        frozen_comparison=frozen_comparison(groups),
        rankings=ranked,
        checks=dict(
            canonical_rows_identical=True,
            model_refits=0,
            network_calls=0,
            bootstrap_draws=DRAWS,
            bootstrap_clusters=dict(
                temporal=int(reference_frames["temporal_validation"].huc.str[:4].nunique()),
                geographic=int(reference_frames["geographic_validation"].huc.str[:4].nunique()),
                pooled=int(reference_pooled_frame.huc.str[:4].nunique()),
            ),
            mismatch_policy="Benchmark metrics are recomputed from per-row artifacts; frozen-record differences are listed and not repaired.",
            source_metric_reproduction=source_metric_reproduction(candidates, candidate_index, groups),
        ),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    destination = out_dir / "benchmark.json"
    destination.write_text(json.dumps(json_value(result), indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR, help="directory for benchmark.json")
    arguments = parser.parse_args()
    result = build_benchmark(arguments.out_dir)
    frozen = result["frozen_candidate"]
    temporal = frozen["metrics"]["temporal_validation"]["own"]["metrics"]
    geographic = frozen["metrics"]["geographic_validation"]["own"]["metrics"]
    print("benchmark written to", arguments.out_dir / "benchmark.json")
    print("groups", len(result["candidate_groups"]), "candidate_index", len(result["candidate_index"]))
    print("frozen own temporal", temporal["detected_events"], temporal["events"], temporal["ap"])
    print("frozen own geographic", geographic["detected_events"], geographic["events"], geographic["ap"])
    print("dominators", sum(1 for row in result["frozen_comparison"] if row["dominates_frozen"] and not row["is_frozen"]))


if __name__ == "__main__":
    main()
