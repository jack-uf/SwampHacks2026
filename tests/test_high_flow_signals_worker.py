"""Contract test for the subdaily neural worker, run as a subprocess like the pipeline runs it."""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

from scripts.hydrology_research import ROOT


def test_worker_scores_every_partition_with_missing_cells(tmp_path):
    generator = np.random.default_rng(3)
    rows = []
    for partition, count in [('inner_train', 900), ('inner_validation', 120),
                             ('calibration', 200), ('threshold', 200),
                             ('temporal_validation', 150), ('geographic_validation', 150)]:
        frame = pd.DataFrame({name: generator.normal(size=count) for name in ['q_2', 'rise_6', 'rain_24_lag6']})
        frame['target'] = (generator.random(count) < .2).astype(int)
        frame['force_rain'] = np.where(generator.random(count) < .25, '', frame.rain_24_lag6)
        frame['partition'] = partition
        frame['station'] = '00000001'
        frame['decision_time'] = pd.date_range('2020-01-01', periods=count, freq='6h', tz='UTC').astype(str)
        rows.append(frame)
    source = tmp_path/'table.csv.gz'
    pd.concat(rows, ignore_index=True).to_csv(source, index=False, compression={'method': 'gzip', 'mtime': 0})
    job = dict(name='worker_test', features=['q_2', 'rise_6', 'rain_24_lag6', 'force_rain'], seed=42,
        epochs=2, patience=1, input=str(source), output=str(tmp_path/'logits.csv.gz'),
        bundle=str(tmp_path/'network.pt'))
    job_path = tmp_path/'job.json'
    job_path.write_text(json.dumps(job))
    environment = dict(os.environ, OMP_NUM_THREADS='2', KMP_DUPLICATE_LIB_OK='TRUE', MPLCONFIGDIR=str(tmp_path))
    result = subprocess.run([sys.executable, '-m', 'scripts.high_flow_signals_worker', str(job_path)],
        cwd=ROOT, capture_output=True, text=True, env=environment)
    assert result.returncode == 0, result.stderr[-800:]
    logits = pd.read_csv(job['output'], dtype={'station': str}, keep_default_na=False)
    assert set(logits.partition) == {'calibration', 'threshold', 'temporal_validation', 'geographic_validation'}
    assert np.isfinite(logits.logit.to_numpy(float)).all()
    assert logits.groupby('partition').size().eq(200).all() or len(logits) >= 600
    assert (tmp_path/'network.pt').exists()
