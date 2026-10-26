"""Inspect a pinned author-repository archive without importing its code."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--source", required=True, type=Path)
parser.add_argument("--paper", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
args = parser.parse_args()
roots = set()
syntax_errors = []
for file in args.source.rglob("*.py"):
    try:
        tree = ast.parse(file.read_text(encoding="utf-8-sig"))
    except (SyntaxError, UnicodeError) as error:
        syntax_errors.append({"file": str(file.relative_to(args.source)), "error": str(error)})
        continue
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        if isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
report = {"paper_doi": "10.1016/j.eswa.2025.126483",
          "paper_sha256": hashlib.sha256(args.paper.read_bytes()).hexdigest(),
          "repository": "https://github.com/derpberk/ModelGuidedPatrolling",
          "commit": "e951a751012b52b9271d60165fe3daf951b469e5",
          "correspondence_status": "candidate_implementation_not_verified_final_paper_release",
          "import_roots": sorted(roots), "syntax_errors": syntax_errors,
          "checkpoint_files": [str(p.relative_to(args.source)) for p in args.source.rglob("*") if p.suffix in (".pt", ".pth", ".ckpt")],
          "dependency_manifests": [str(p.relative_to(args.source)) for p in args.source.rglob("*") if p.name.startswith("requirements") or p.name in ("pyproject.toml", "environment.yml")],
          "issues": ["no_checkpoint_files_in_pinned_archive", "Train.py_hardcodes_N_and_environment_seed",
                     "VAEUnet_requires_pretrained_VGG16_and_external_checkpoints",
                     "paper_KL_direction_and_candidate_code_direction_differ",
                     "candidate_code_logvar_conversion_differs_between_sampling_and_KL",
                     "no_LICENSE_file_in_pinned_archive; external_source_not_redistributed_in_project_commit"],
          "original_code_executed": False}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps({"checkpoints": len(report["checkpoint_files"]), "syntax_errors": len(syntax_errors), "output": str(args.output)}))
