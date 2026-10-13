"""Execute both notebooks using the current environment; record actual outcomes."""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JUPYTER_DATA = Path('/tmp/bayan-jupyter-data')
os.environ['JUPYTER_DATA_DIR'] = str(JUPYTER_DATA)
os.environ['JUPYTER_RUNTIME_DIR'] = '/tmp/bayan-jupyter-runtime'
os.environ['IPYTHONDIR'] = '/tmp/bayan-ipython'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/bayan-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME', '/workspace/.cache')

import nbformat
from nbclient import NotebookClient


def main():
    kernel = JUPYTER_DATA / 'kernels' / 'bayan-local'
    kernel.mkdir(parents=True, exist_ok=True)
    (kernel / 'kernel.json').write_text(json.dumps({'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
                                                  'display_name': 'Bayan local validation', 'language': 'python'}))
    results = []
    for name in ['bayan_NLP_Wadeen_Khalid_Al_adwani.ipynb', 'notebooks/01_exploration.ipynb']:
        notebook = nbformat.read(ROOT / name, as_version=4)
        NotebookClient(notebook, timeout=180, kernel_name='bayan-local', resources={'metadata': {'path': str(ROOT)}}).execute()
        nbformat.write(notebook, ROOT / name)
        count = sum(cell.cell_type == 'code' for cell in notebook.cells)
        assert all(cell.execution_count is not None for cell in notebook.cells if cell.cell_type == 'code')
        assert not any(output.output_type == 'error' for cell in notebook.cells if cell.cell_type == 'code' for output in cell.outputs)
        results.append({'notebook': name, 'executed_code_cells': count, 'errors': 0, 'status': 'passed'})
    (ROOT / 'reports/notebook_validation.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
