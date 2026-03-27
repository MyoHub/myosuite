"""Execute a single notebook and normalize outputs for nbformat validation."""
import sys
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


def normalize_cell_outputs(cell):
    """Ensure cell outputs have required fields for nbformat 4.5+."""
    for out in cell.get("outputs", []):
        if out.get("output_type") == "stream" and "name" not in out:
            out["name"] = "stdout"
        if out.get("output_type") in ("display_data", "execute_result") and "metadata" not in out:
            out["metadata"] = {}
        if out.get("output_type") == "execute_result" and "execution_count" not in out:
            out["execution_count"] = None


class NormalizingExecutePreprocessor(ExecutePreprocessor):
    """Execute and normalize outputs after each cell so validation passes."""

    def preprocess_cell(self, cell, resources, index):
        cell, resources = super().preprocess_cell(cell, resources, index)
        normalize_cell_outputs(cell)
        return cell, resources


def main():
    if len(sys.argv) != 2:
        print("Usage: execute_tutorial.py <notebook path>", file=sys.stderr)
        sys.exit(2)
    path = Path(sys.argv[1]).resolve()
    if not path.exists():
        print(f"Not found: {path}", file=sys.stderr)
        sys.exit(1)
    nb = nbformat.read(path, as_version=4)
    for cell in nb.cells:
        normalize_cell_outputs(cell)
    ep = NormalizingExecutePreprocessor(kernel_name="myosuite_uv", timeout=600)
    ep.preprocess(nb, {"metadata": {"path": str(path.parent)}})
    out_path = path.parent / (path.stem + ".nbconvert.ipynb")
    nbformat.write(nb, out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
