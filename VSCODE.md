# VS Code beginner guide — run SupplyPrescript click-by-click

This guide assumes you have **never** used VS Code with a Python project.
Follow the steps **in order** the first time. Later days you only need
**Part D** (start the API) and the demos.

> **Important:** Always open the folder named `Supply-Prescript` (the repo
> root). Do **not** open `week1`, `week2`, or `week3` by themselves.

---

## What you will set up

| Piece | Why you need it |
|---|---|
| VS Code (or Cursor) | Editor + Run and Debug buttons |
| Python 3.10+ | Runs the code |
| `.venv` folder | Private install of project libraries |
| Python extension | Lets VS Code find Python and press F5 |
| Trained model files | So `/predict` and the dashboard work |

---

## Part A — Install the tools (once per computer)

### A1. Install Python

1. Go to https://www.python.org/downloads/
2. Download **Python 3.10 or newer**.
3. Run the installer.
4. On Windows: tick **“Add python.exe to PATH”** before clicking Install.
5. Check it worked:

Open a terminal (any) and type:

```text
python --version
```

You should see something like `Python 3.12.3`.  
If Windows says `python` is not found, try `py --version`.

### A2. Install VS Code (or Cursor)

- VS Code: https://code.visualstudio.com/
- Or use **Cursor** (same steps; menus look almost identical)

### A3. Get the project folder on your machine

If you already cloned with Git:

```text
cd Desktop
git clone https://github.com/bhattchintan8540-cmd/Supply-Prescript.git
cd Supply-Prescript
git checkout main
git pull origin main
```

If you downloaded a ZIP: unzip it, then rename/move so you have a folder
like:

```text
C:\Users\ACER\Desktop\Supply-Prescript
```

---

## Part B — Open the project correctly

1. Start **VS Code**.
2. Menu: **File → Open Folder…**  
   (Mac: **File → Open…**)
3. Select the folder **`Supply-Prescript`** (the one that contains
   `README.md`, `week1`, `week2`, `week3`, `requirements.txt`).
4. Click **Select Folder** / **Open**.
5. If VS Code asks **“Do you trust the authors of the files in this folder?”**
   → click **Yes, I trust the authors**.

### Check you opened the right place

Look at the **Explorer** (left sidebar, top icon that looks like files).
You must see all of these names at the top level:

```text
.vscode
week1
week2
week3
week4
week5
README.md
requirements.txt
VSCODE.md
```

If you only see files from inside `week3`, you opened the wrong folder.
Close the window and repeat **File → Open Folder…** on the parent.

---

## Part C — One-time project setup (do this once)

### C1. Install recommended extensions

1. When VS Code opens the folder, it may show a popup:
   **“This workspace has extension recommendations.”**
2. Click **Install All** (or **Install**).
3. If no popup appeared:
   - Click the **Extensions** icon in the left sidebar  
     (four squares, or press `Ctrl+Shift+X` / Mac `Cmd+Shift+X`)
   - In the search box type: `@recommended`
   - Install at least:
     - **Python** (Microsoft)
     - **Pylance**
     - **Jupyter** (for notebooks)
     - **Python Debugger** / debugpy (often comes with Python)

Wait until installs finish (bottom status bar).

### C2. Create the virtual environment and install libraries

**Option 1 — easiest (Task)**

1. Press `Ctrl+Shift+P` (Mac: `Cmd+Shift+P`) to open the **Command Palette**.
2. Type: `Tasks: Run Task`
3. Press Enter.
4. Choose: **Setup: create venv + install deps**
5. Watch the **Terminal** panel at the bottom.  
   Wait until pip finishes (last lines look like `Successfully installed …`).
6. This can take several minutes the first time (XGBoost / scientific stack).

**Option 2 — type it yourself**

1. Menu: **Terminal → New Terminal**
2. Make sure the prompt shows you are inside `Supply-Prescript`.
3. Run these commands **one at a time**:

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If `python` fails on Windows, try `py -m venv .venv` instead.

### C3. Tell VS Code which Python to use

1. Press `Ctrl+Shift+P` / `Cmd+Shift+P`
2. Type: `Python: Select Interpreter`
3. Press Enter.
4. Pick the one that contains **`.venv`**:
   - Windows: `.\.venv\Scripts\python.exe`
   - macOS/Linux: `./.venv/bin/python`
5. After you pick it, the bottom-right status bar should show a Python
   version path that includes `.venv`.

If `.venv` is missing from the list:

- Click **Enter interpreter path…** → **Find…**
- Browse into `.venv` → `Scripts` (Windows) or `bin` (Mac/Linux) → `python`

### C4. Build data + train the model (required before the API)

The dashboard needs two files under `data/`:

- `shipments.csv`
- `delay_model.joblib`

**Easiest path in VS Code:**

1. Press `Ctrl+Shift+P` / `Cmd+Shift+P`
2. Run **Tasks: Run Task**
3. Choose **Pipeline: data → train**
4. Wait for both scripts to finish in the terminal.  
   You should see messages about writing CSV and saving the model.

**Or use Run and Debug (F5) twice:**

1. Click the **Run and Debug** icon in the left sidebar  
   (play button with a bug), or press `Ctrl+Shift+D` / `Cmd+Shift+D`
2. At the top, open the dropdown next to the green play button.
3. Select **Week1: generate mock data** → click the green ▶ (or press `F5`)
4. Wait until the terminal says it finished / returns to a prompt.
5. Change the dropdown to **Week1: train model** → ▶ / `F5` again.
6. Confirm these files now exist in Explorer under `data/`:
   - `shipments.csv`
   - `delay_model.joblib`
   - `metrics.json`

---

## Part D — Start the app (every time you demo)

1. Left sidebar → **Run and Debug**
2. Dropdown → **API: FastAPI (uvicorn --reload)**
3. Click green ▶ or press `F5`
4. In the terminal you should see something like:

```text
Uvicorn running on http://127.0.0.1:8000
```

5. Open your browser and go to [http://127.0.0.1:8000/ui/](http://127.0.0.1:8000/ui/)
   (or just [http://127.0.0.1:8000](http://127.0.0.1:8000) — it redirects to the dashboard).

6. Optional API docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

7. To stop the server: click the red ■ **Stop** button in the Debug toolbar,
   or focus the terminal and press `Ctrl+C`.

### First demo clicks in the browser

1. On the dashboard, click **Demo A · Reliable / off-peak**
2. Wait for prediction + option cards
3. Click **Execute decision** on one option (for example Air Freight)
4. In the decisions table, click **Log outcome**
5. Accept the suggested cost/delay (or type numbers) → OK
6. Click **Refresh** if needed — Intervention ROI and cost accuracy update

Also try **Demo B** and **Demo C** to show peak season / risky supplier.

---

## Part E — Other useful Run and Debug configs

Same place: **Run and Debug** dropdown → pick one → ▶ / `F5`

| Name in dropdown | When to use | What you should see |
|---|---|---|
| **Week1: generate mock data** | First setup / reset data | Creates `data/shipments.csv` |
| **Week1: explore data** | Want charts for slides | PNGs appear under `docs/figures/` |
| **Week1: train model** | After new data | Updates `delay_model.joblib` + `metrics.json` |
| **Week1: demo model (CLI)** | Live talk without browser | Prints 4 shipment scenarios |
| **Week1: evaluate XGBoost** | Week 5 / ML process story | Confusion-matrix files under `data/ml_evaluation/` |
| **Week2: demo prescribe** | Show expected-cost options in terminal | Prints Air / Secondary / Delay / Optimizer |
| **Week4: retrain** | After some outcomes logged | Retrains only if cost drift is high |
| **Week4: retrain --force** | Force a retrain for the demo | Always refits |
| **Week5: smoke closed loop** | Check the whole loop without browser | Prints `SMOKE OK` |
| **Pytest: all tests** | Before presenting | Many dots then `passed` |
| **Pytest: current file** | Debugging one test file | Runs only the open `test_*.py` |

---

## Part F — Tasks (run without the debugger)

1. Menu: **Terminal → Run Task…**  
   or Command Palette → `Tasks: Run Task`
2. Pick a task:

| Task | Purpose |
|---|---|
| **Setup: create venv + install deps** | First-time install |
| **Pipeline: data → train** | Generate CSV then train (best first build) |
| **Week1: generate mock data** | Data only |
| **Week1: explore data (EDA charts)** | Charts only |
| **Week1: train delay model** | Train only |
| **Week1: evaluate XGBoost (confusion matrix)** | Week 5 evaluation |
| **Week4: retrain if drift** | Continuous learning |
| **Week5: smoke closed loop** | End-to-end check |
| **Tests: pytest** | Run the test suite |

---

## Part G — Run tests from the Testing sidebar

1. Click the **Testing** icon in the left sidebar (beaker / flask).
2. If asked to configure: choose **pytest**, then the workspace folder `.`
3. Click **Run Tests** (play icon).
4. All tests green = good to present.

---

## Part H — Open a notebook

1. In Explorer, open `notebooks/01_exploratory_analysis.ipynb`
2. Top-right of the notebook → **Select Kernel**
3. Choose the same `.venv` Python
4. Run cells with the ▶ button on each cell, or **Run All**

Also available: `02_dataset_analysis.ipynb`, `03_xgboost_ml_process.ipynb`

---

## Part I — Open the slide deck

In Explorer, right-click `docs/presentation/slides.html` →  
**Reveal in File Explorer** / **Reveal in Finder**, then double-click it  
**or** open it from a browser with File → Open File.

---

## Beginner checklist (print this)

- [ ] Python installed (`python --version` works)
- [ ] Opened folder `Supply-Prescript` (see `week1` + `README.md`)
- [ ] Installed Python + Jupyter extensions
- [ ] Ran **Setup: create venv + install deps** (or pip install)
- [ ] Selected `.venv` interpreter
- [ ] Ran **Pipeline: data → train** (or generate + train)
- [ ] Confirmed `data/shipments.csv` and `data/delay_model.joblib` exist
- [ ] Started **API: FastAPI (uvicorn --reload)**
- [ ] Opened http://127.0.0.1:8000/ui/
- [ ] Clicked Demo A / B / C and logged one outcome

---

## Common beginner mistakes (and exact fixes)

### 1. `ModuleNotFoundError: No module named 'week1'`

**Cause:** Wrong folder opened, or interpreter is not `.venv`.

**Fix:**

1. **File → Open Folder…** → choose `Supply-Prescript` root again
2. `Python: Select Interpreter` → `.venv`
3. Re-run the launch config

### 2. API says model not found / HTTP 503

**Cause:** You never trained.

**Fix:** Run **Week1: train model** (and generate data first if CSV is missing),
then start the API again.

### 3. `python` is not recognized (Windows)

**Fix:** Reinstall Python with **Add to PATH**, or use `py` instead of `python`.
In VS Code still select `.venv\Scripts\python.exe` as the interpreter.

### 4. Terminal says `source` is not recognized

**Cause:** That command is for Mac/Linux.

**Fix (Windows PowerShell):**

```powershell
.\.venv\Scripts\activate
```

### 5. Port 8000 already in use

**Fix:** Stop the old debug session (red ■), or close the other terminal
running uvicorn. Then start **API: FastAPI** again.

### 6. Dashboard form fails / “Failed to fetch”

**Cause:** API is not running, or you opened the HTML file as a local file
without the server.

**Fix:** Start **API: FastAPI** and use http://127.0.0.1:8000/ui/  
(not a `file:///.../index.html` path).

### 7. Extensions installed but Run and Debug list is empty

**Fix:** Confirm `.vscode/launch.json` exists in Explorer. Reload the window:
Command Palette → `Developer: Reload Window`.

### 8. pip install is very slow or fails on XGBoost

**Fix:** Use Python 64-bit (not 32-bit). Retry the setup task. On corporate
networks you may need a different pip index — ask your instructor.

---

## After the first setup — short daily path

1. Open `Supply-Prescript` in VS Code  
2. Run and Debug → **API: FastAPI (uvicorn --reload)** → ▶  
3. Browser → http://127.0.0.1:8000/ui/  
4. Optional: **Week1: demo model (CLI)** in another debug run for the talk

That’s the whole loop.
