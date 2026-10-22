#!/usr/bin/env python3
"""Verify the notebook, sources, result CSVs, revision outputs and generated manuscript inputs.

Run from the repository root:  python scripts/verify_artifacts.py
Writes docs/artifact_validation.json and exits with an AssertionError on any failure.
"""
from pathlib import Path
import hashlib, json, sys

import nbformat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
NOTEBOOK = "SciGuard_JMLR_Reproducible_Analysis_MIMIC_IV_Clean.ipynb"
# Files edited after the archived full run.  They are not used by any archived section, and
# the edit is reported rather than hidden.  Any other source mismatch fails verification.
KNOWN_POST_RUN_EDITS = {
    "src/sciguard/mimic.py": ("edited for the credentialed MIMIC-IV execution after the archived "
                              "public/synthetic run; not imported by any archived section"),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(name):
    return name.replace("\\", "/")          # manifests written on Windows use backslashes


report = {"status": "running"}

# ---- canonical notebook ------------------------------------------------------------------------
candidates = [ROOT / NOTEBOOK] + sorted(ROOT.glob("SciGuard_JMLR_Reproducible_Analysis*.ipynb"))
nb_path = next(p for p in candidates if p.is_file())
nb = nbformat.read(nb_path, 4); nbformat.validate(nb)
code = [c for c in nb.cells if c.cell_type == "code"]
assert nb.metadata.sciguard.execution_status == "completed"
assert not any(o.output_type == "error" for c in code for o in c.outputs), "notebook contains error outputs"
headers = [c.source.splitlines()[0] for c in nb.cells if c.cell_type == "markdown" and c.source.startswith("## ")]
revision_index = next(i for i, c in enumerate(nb.cells) if c.cell_type == "markdown" and c.source.startswith("## 13. Results revision"))
revision_code = [c for c in nb.cells[revision_index:] if c.cell_type == "code"]
assert revision_code and all(c.outputs for c in revision_code), "results-revision cells lack outputs"
report["notebook"] = {"file": nb_path.name, "code_cells": len(code), "sections": len(headers),
                      "results_revision_code_cells_with_outputs": len(revision_code),
                      "embedded_pngs": sum("image/png" in o.get("data", {}) for c in code for o in c.outputs),
                      "metadata": dict(nb.metadata.sciguard)}

# ---- archived full run -------------------------------------------------------------------------
manifest = json.loads((ROOT / "results/full/run_manifest.json").read_text())
assert str(manifest["status"]).startswith("completed")
assert manifest["external_benchmark"] == "completed_credentialed_data"
post_run_edits = {}
for name, expected in manifest["source_sha256"].items():
    path = rel(name)
    if sha(ROOT / path) != expected:
        assert path in KNOWN_POST_RUN_EDITS, f"source changed after the archived run: {path}"
        post_run_edits[path] = KNOWN_POST_RUN_EDITS[path]
for name, expected in manifest["input_sha256"].items():
    assert sha(ROOT / rel(name)) == expected, name
for name, expected in manifest["output_sha256"].items():
    assert sha(ROOT / "results/full" / name) == expected, name
report["full_run"] = {"status": manifest["status"], "external_benchmark": manifest["external_benchmark"],
                      "source_files_verified": len(manifest["source_sha256"]) - len(post_run_edits),
                      "sources_edited_after_run": post_run_edits,
                      "result_csvs_verified": len(manifest["output_sha256"])}

# ---- revision outputs --------------------------------------------------------------------------
from sciguard.revision import REVISION_SECTIONS  # noqa: E402
rev = json.loads((ROOT / "results/revision/revision_manifest.json").read_text())
assert rev["status"] == "completed"
assert set(REVISION_SECTIONS) <= set(rev["sections"]), "revision section missing"
assert all(v["status"] == "completed" for v in rev["sections"].values())
for name, expected in rev["output_sha256"].items():
    assert sha(ROOT / "results/revision" / name) == expected, name
for name, expected in rev["input_sha256"].items():
    assert sha(ROOT / rel(name)) == expected, name
for name, expected in rev["source_sha256"].items():
    assert sha(ROOT / rel(name)) == expected, f"source changed after the revision run: {name}"
report["revision_run"] = {"sections": sorted(rev["sections"]), "python": rev["python"],
                          "platform": rev["platform"], "result_csvs_verified": len(rev["output_sha256"]),
                          "inputs_verified": len(rev["input_sha256"]), "sources_verified": len(rev["source_sha256"])}

# ---- generated manuscript inputs -------------------------------------------------------------
prov = json.loads((ROOT / "manuscript/generated/table_provenance.json").read_text())
for folder, key in [("results/full", "source_csv_sha256"), ("results/revision", "revision_csv_sha256"),
                    ("results/mimic_local", "mimic_local_csv_sha256")]:
    for name, expected in prov.get(key, {}).items():
        assert sha(ROOT / folder / name) == expected, f"{folder}/{name}"
for name, expected in prov.get("generated_files", {}).items():
    assert sha(ROOT / "manuscript/generated" / name) == expected, name
report["manuscript_inputs"] = {"generated_files_verified": len(prov.get("generated_files", {})),
                               "mimic_operating_table": (ROOT / "manuscript/generated/mimic_operating.tex").is_file()}

# ---- PDFs and LaTeX log ------------------------------------------------------------------------
try:
    from pypdf import PdfReader
    pdfs = sorted((ROOT / "figures/full").glob("*.pdf")) + sorted((ROOT / "figures/revision").glob("*.pdf"))
    pdfs += [p for p in [ROOT / "manuscript/main_jmlr.pdf"] if p.is_file()]
    report["pdf_pages"] = {str(p.relative_to(ROOT)): len(PdfReader(p).pages) for p in pdfs}
except ImportError:
    report["pdf_pages"] = "pypdf not installed"
log_path = ROOT / "manuscript/main_jmlr.log"
if log_path.is_file():
    log = log_path.read_text(errors="replace")
    assert "Overfull" not in log, "overfull box in manuscript log"
    assert "undefined" not in log.lower(), "undefined reference or citation in manuscript log"
    assert "Rerun to get cross-references right" not in log, "LaTeX needs another pass"
    report["latex_log"] = "no errors, overfull boxes, or undefined references"

report["clinical_data_execution"] = manifest["external_benchmark"]
report["status"] = "passed"
(ROOT / "docs/artifact_validation.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
