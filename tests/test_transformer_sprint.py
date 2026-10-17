"""Experiment boundaries; fixtures are synthetic and never enter training files.

torch and the boosted libraries cannot share one process here, so the worker runs
in a subprocess in this test exactly as it does in the experiment.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from scripts import transformer_sprint
from scripts.transformer_sprint import matched_budget_threshold, transformer_gate_failures

FT_FRAMES_SCRIPT = '''
import json
import pathlib
import sys

import numpy as np

frames_dir, out_dir = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
frames_dir.mkdir(parents=True, exist_ok=True)
out_dir.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)
features = ["feature_%d" % i for i in range(6)]
rows = rng.normal(size=(400, 6))
label = (rows[:, 0] + rng.normal(size=400) * .5 > 0).astype(int)
frame = __import__("pandas").DataFrame(rows, columns=features)
frame["exceedance"] = label
frame.iloc[:300].to_csv(frames_dir / "train.csv", index=False)
frame.iloc[300:].to_csv(frames_dir / "calibration.csv", index=False)
frame.iloc[300:].to_csv(frames_dir / "validation.csv", index=False)
(frames_dir / "features.json").write_text(json.dumps({"features": features}))
(frames_dir / "splits.json").write_text(json.dumps({"splits": ["validation"]}))
'''


def run_worker(name, frames, out, reuse=False, models=None):
    command = [sys.executable, '-m', 'scripts.transformer_worker', name, str(frames), str(out)]
    if reuse:
        command.append('--reuse')
    if models:
        command += ['--models', str(models)]
    return subprocess.run(command, capture_output=True, text=True, cwd=transformer_sprint.ROOT,
                          env=dict(os.environ, PYTHONPATH=str(transformer_sprint.ROOT), HF_HUB_OFFLINE='1'))


def test_worker_trains_ft_transformer_deterministically(tmp_path):
    frames, out = tmp_path / 'frames', tmp_path / 'out'
    setup = subprocess.run([sys.executable, '-c', FT_FRAMES_SCRIPT, str(frames), str(out)], capture_output=True, text=True,
                           cwd=transformer_sprint.ROOT, env=dict(os.environ, PYTHONPATH=str(transformer_sprint.ROOT)))
    assert setup.returncode == 0, setup.stderr[-500:]
    first = run_worker('ft_transformer_extended', frames, out)
    assert first.returncode == 0, first.stderr[-800:]
    probabilities = np.load(out / 'ft_transformer_extended.validation.npy')
    assert probabilities.shape == (100,) and np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    meta = json.loads((out / 'ft_transformer_extended.meta.json').read_text())
    assert 1 <= meta['epochs_trained'] <= 60
    reused = run_worker('ft_transformer_extended', frames, tmp_path / 'reused', reuse=True, models=out)
    assert reused.returncode == 0, reused.stderr[-800:]
    assert np.array_equal(probabilities, np.load(tmp_path / 'reused' / 'ft_transformer_extended.validation.npy'))


def test_matched_budget_threshold_equals_the_rule_false_alert_count():
    y = np.array([1, 1] + [0] * 8)
    scores = np.array([.9, .8, .7, .6, .5, .4, .3, .2, .1, .05])
    threshold = matched_budget_threshold(scores, y, .25)
    assert threshold == .6
    assert int(np.sum((scores >= threshold) & (y == 0))) == 2
    assert matched_budget_threshold(scores, y, 0) > .9
    assert int(np.sum(scores >= matched_budget_threshold(scores, y, 0))) == 0
    assert matched_budget_threshold(scores, y, 1.0) == .05
    with pytest.raises(ValueError):
        matched_budget_threshold(scores, np.ones(10), .25)


def test_extension_rejects_days_before_the_fresh_window(tmp_path, monkeypatch):
    columns = {name: [1.0] for name in transformer_sprint.EXTENDED}
    stale = pd.DataFrame(dict(station=['ABQ'], target_date=['2026-06-30'], role=['development'], **columns))
    path = tmp_path / 'dataset-extension.csv.gz'
    stale.to_csv(path, index=False)
    monkeypatch.setattr(transformer_sprint, 'OUT', tmp_path)
    monkeypatch.setattr(transformer_sprint.wind_sprint, 'load_dataset',
                        lambda: pd.DataFrame(dict(station=['ABQ'], target_date=['2026-06-30'], role=['development'])))
    with pytest.raises(ValueError, match='fresh window'):
        transformer_sprint.extension_frames()
    fresh = stale.copy()
    fresh['target_date'] = '2026-07-01'
    fresh.to_csv(path, index=False)
    frozen, frame = transformer_sprint.extension_frames()
    assert list(frame.target_date) == ['2026-07-01']
    unknown = fresh.copy()
    unknown['station'] = 'ZZZ'
    unknown.to_csv(path, index=False)
    with pytest.raises(ValueError, match='outside the frozen protocol'):
        transformer_sprint.extension_frames()


def test_gate_adds_the_matched_budget_recall_condition():
    updated = dict(positives=30, alert_rate=.12, recall=.6, false_alerts_per_100_station_days=3.)
    baseline = dict(recall=.5, false_alerts_per_100_station_days=5.)
    intervals = dict(available=True, delta_ap_95=[.01, .08])
    stations = {'JFK': dict(updated=dict(positives=10, recall=.6), baseline=dict(recall=.5))}
    assert transformer_gate_failures(updated, baseline, intervals, stations, True, dict(recall=.55)) == []
    failures = transformer_gate_failures(updated, baseline, intervals, stations, True, dict(recall=.4))
    assert failures == ['Matched-budget recall below the forecast rule']
    inherited = transformer_gate_failures(dict(updated, recall=.4), baseline, intervals, stations, True, dict(recall=.6))
    assert any('Recall below 30%' in failure for failure in inherited)


def test_frozen_selection_rejects_changed_code_and_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(transformer_sprint, 'ROOT', tmp_path)
    experiment = tmp_path / 'experiment'
    experiment.mkdir()
    monkeypatch.setattr(transformer_sprint, 'OUT', experiment)
    (experiment / 'probabilities').mkdir()
    (experiment / 'probabilities' / 'logistic_extended.model.joblib').write_text('frozen model')
    document = tmp_path / 'docs/TRANSFORMER_SPRINT_PROTOCOL.md'
    document.parent.mkdir()
    document.write_text('frozen')
    monkeypatch.setattr(transformer_sprint, 'PROTOCOL_DOCUMENT', document)
    wind = tmp_path / 'wind'
    wind.mkdir()
    monkeypatch.setattr(transformer_sprint.wind_sprint, 'OUT', wind)
    artifacts = [(wind / 'training.csv.gz', 'dataset_sha256'), (wind / 'protocol.json', 'protocol_sha256'),
                 (document, 'protocol_document_sha256'), (tmp_path / 'scripts/transformer_sprint.py', 'code_sha256'),
                 (tmp_path / 'scripts/transformer_worker.py', 'worker_sha256')]
    selection = dict(versions={}, checkpoint_sha256=transformer_sprint.CHECKPOINT_SHA256)
    for path, field in artifacts:
        path.parent.mkdir(exist_ok=True)
        path.write_text('frozen artifact')
        selection[field] = transformer_sprint.digest(path)
    selection['artifacts_sha256'] = transformer_sprint.artifacts_digest()
    (experiment / 'selection.json').write_text(json.dumps(selection))
    assert transformer_sprint.frozen_selection() == selection
    (experiment / 'probabilities' / 'logistic_extended.model.joblib').write_text('changed')
    with pytest.raises(ValueError, match='candidate artifacts changed'):
        transformer_sprint.frozen_selection()
    (experiment / 'probabilities' / 'logistic_extended.model.joblib').write_text('frozen model')
    selection['checkpoint_sha256'] = '0' * 64
    (experiment / 'selection.json').write_text(json.dumps(selection))
    with pytest.raises(ValueError, match='checkpoint'):
        transformer_sprint.frozen_selection()
    selection['checkpoint_sha256'] = transformer_sprint.CHECKPOINT_SHA256
    (experiment / 'selection.json').write_text(json.dumps(selection))
    (tmp_path / 'scripts/transformer_worker.py').write_text('changed after selection')
    with pytest.raises(ValueError, match='Frozen selection input changed'):
        transformer_sprint.frozen_selection()


def test_interrupted_evaluation_cannot_be_restarted(tmp_path, monkeypatch):
    monkeypatch.setattr(transformer_sprint, 'OUT', tmp_path)
    monkeypatch.setattr(transformer_sprint, 'frozen_selection', lambda: {})
    monkeypatch.setattr(transformer_sprint, 'verify_checkpoint', lambda: None)
    (tmp_path / 'selection.json').write_text('{}')
    def interrupted():
        raise RuntimeError('interrupted')
    monkeypatch.setattr(transformer_sprint, 'selection_inputs', interrupted)
    with pytest.raises(RuntimeError, match='interrupted'):
        transformer_sprint.evaluate()
    with pytest.raises(FileExistsError):
        transformer_sprint.evaluate()

