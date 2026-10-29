"""Require a real Torch runtime and prohibit skipped tests in the final suite."""
import argparse
import io
import json
from pathlib import Path
import time
import unittest

import torch

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
args.output.mkdir(exist_ok=False);torch.set_num_threads(1)
stream=io.StringIO();started=time.perf_counter()
suite=unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parents[1]/'tests'))
result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
(args.output/'full-tests.txt').write_text(stream.getvalue(),encoding='utf-8')
(args.output/'full-tests.json').write_text(json.dumps({'torch':torch.__version__,'tests':result.testsRun,
    'skipped':len(result.skipped),'failures':len(result.failures),'errors':len(result.errors),
    'success':result.wasSuccessful(),'seconds':time.perf_counter()-started},indent=2)+'\n')
if result.skipped or not result.wasSuccessful():raise RuntimeError('Full actual-Torch suite failed or skipped')
print('ALL_TORCH_TESTS_PASSED',result.testsRun,flush=True)
