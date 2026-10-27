"""Read unchanged JSON evidence, optionally stored as lossless gzip."""
import gzip
import json
from pathlib import Path


def read_json(path):
    path=Path(path)
    if path.exists():
        return json.loads(path.read_text())
    compressed=path.with_name(path.name+'.gz')
    if compressed.exists():
        with gzip.open(compressed,'rt',encoding='utf-8') as handle:return json.load(handle)
    raise FileNotFoundError(path)
