"""Re-run the immutable research baseline in temporary storage; never serve old artifacts."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import tempfile
from scripts.common import ROOT


def main():
    archive=ROOT/'experiments/baseline'
    manifest=json.loads((archive/'manifest.json').read_text())
    for name,expected in manifest['files'].items():
        if hashlib.sha256((archive/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('Baseline archive checksum mismatch: '+name)
    spec=importlib.util.spec_from_file_location('baseline_experiment',archive/'scripts/train.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    expected=json.loads((archive/'models/evaluation.json').read_text())
    with tempfile.TemporaryDirectory(prefix='planit-baseline-') as directory:
        root=Path(directory);(root/'data').mkdir();(root/'docs').mkdir()
        shutil.copyfile(archive/'data/training.csv',root/'data/training.csv')
        # The archived experiment resolves only its frozen data and feature order.
        module.ROOT=root;module.FEATURES=expected['features'];module.main()
        actual=json.loads((root/'models/evaluation.json').read_text())
    count=0
    def compare(a,b,path=''):
        nonlocal count
        if isinstance(a,dict):
            assert set(a)==set(b),path
            for key in a:
                if key!='created_at':compare(a[key],b[key],path+'/'+key)
        elif isinstance(a,list):
            assert len(a)==len(b),path
            for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+'/'+str(i))
        elif isinstance(a,(int,float)) and not isinstance(a,bool):
            assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-12),(path,a,b)
            count+=1
        else:assert a==b,(path,a,b)
    compare(expected,actual)
    result=dict(baseline_commit=manifest['commit'],numeric_fields_compared=count,matched=True,
                tolerance='relative 1e-10; absolute 1e-12; creation timestamp excluded',archive_checksums_verified=True)
    (ROOT/'output').mkdir(exist_ok=True)
    (ROOT/'output/baseline-reproduction.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
