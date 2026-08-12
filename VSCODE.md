# Run SupplyPrescript in VS Code / Cursor

Open the **repo root** as the workspace folder (not a single week subfolder).

## One-time setup

1. Install recommended extensions when prompted (Python, Pylance, Jupyter), or:
   `Extensions` → search `@recommended`
2. Create the venv and install deps:
   - **Command Palette** → `Tasks: Run Task` → **Setup: create venv + install deps**
   - Or terminal:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
```

3. Select the interpreter: `Python: Select Interpreter` → choose `.venv`
   - Linux/macOS: `.venv/bin/python`
   - Windows: `.venv\Scripts\python.exe`

## Run / debug (F5)

Open **Run and Debug** and pick a configuration:

| Configuration | What it does |
|---|---|
| **API: FastAPI (uvicorn --reload)** | Starts the dashboard/API → http://127.0.0.1:8000/ui/ |
| **Week1: generate mock data** | Writes `data/shipments.csv` |
| **Week1: train model** | Fits XGBoost → `data/delay_model.joblib` |
| **Week1: demo model (CLI)** | Four-scenario delay demo |
| **Week1: evaluate XGBoost** | Confusion matrix + verdict |
| **Week2: demo prescribe** | Expected-cost options in the terminal |
| **Week4: retrain** / **--force** | Drift-triggered retrain |
| **Week5: smoke closed loop** | Full loop without the browser |
| **Pytest: all tests** | Test suite |
| **Pytest: current file** | Tests in the open file |

First-time demo path: run **Week1: generate mock data**, then **Week1: train model**, then **API: FastAPI**.

## Tasks (no debugger)

`Terminal` → `Run Task…`

- **Pipeline: data → train** (default build)
- **Tests: pytest** (default test)
- Individual week scripts (EDA, evaluate, smoke, retrain)

## Tests in the UI

Testing sidebar → configure pytest if asked → **Run All Tests**.  
Settings already enable pytest at the workspace root.

## Notebooks

Open `notebooks/01_exploratory_analysis.ipynb` (and 02/03) with the Jupyter extension.  
Kernel = the same `.venv` interpreter. Notebook root is the repo folder.

## Common mistakes

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: week1` | Workspace folder must be the repo root; `PYTHONPATH` is set in launch/tasks |
| Wrong interpreter | `Python: Select Interpreter` → `.venv` |
| API 503 / model missing | Run **Week1: train model** before the API config |
| Opened `week3/` as the folder | Re-open the parent `Supply-Prescript` folder |
