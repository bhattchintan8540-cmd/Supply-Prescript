# Windows setup (PowerShell)

For machines like `C:\Users\ACER\Desktop\Supply-Prescript`.

## One-time setup

```powershell
cd C:\Users\ACER\Desktop\Supply-Prescript
git checkout main
git pull origin main

python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## VS Code / Cursor (recommended)

1. **File → Open Folder** → `Supply-Prescript` (the repo root).
2. Install recommended extensions when prompted (Python, Jupyter).
3. `Ctrl+Shift+P` → **Python: Select Interpreter** → `.venv\Scripts\python.exe`.
4. **Run and Debug** → pick **API: FastAPI (uvicorn --reload)** (train the model first).
5. Or **Terminal → Run Task…** → **Pipeline: data → train**.

Full list of launch configs and tasks: [VSCODE.md](VSCODE.md).

## Train the model

```powershell
python week1\generate_mock_data.py
python week1\train_model.py
```

## Presentation demos

**Terminal (show the model):**

```powershell
python week1\demo_model.py
python week2\demo_prescribe.py
```

**Browser (full closed loop):**

```powershell
uvicorn week3.main:app --reload
```

Then open http://127.0.0.1:8000/ui/ and click **Demo A / B / C**.

**Slides:**

```powershell
start docs\presentation\slides.html
```

## Common Windows fixes

| Problem | Fix |
|---|---|
| `can't open file ... week1\demo_model.py` | You are not in the repo folder — `cd` into `Supply-Prescript` first |
| `source` is not recognized | Use `.\.venv\Scripts\activate` (not `source`) |
| `python` not found | Install Python 3.10+ and tick “Add python.exe to PATH” |
| Port 8000 in use | `uvicorn week3.main:app --reload --port 8001` |
