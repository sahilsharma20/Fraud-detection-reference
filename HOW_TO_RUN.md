# ▶️ How to Run This Project (Simple Guide)

This is the **easy** guide. If you just cloned this repo and want to see it working, follow the steps
below. No machine-learning knowledge needed.

---

## What does this project do?
You open a web page, type in some details about an insurance claim, and click a button. The app tells
you:
1. **Is this claim likely fraud?** (✅ Genuine, or 🚩 Needs Review)
2. **If genuine — how much money** the claim is likely worth.
3. **Why** it decided that (the top reasons).

The trained models are **already included** in this repo, so you do **not** need to train anything.
Just install and run.

---

## Step 1 — Install Python (one time)
You need **Python 3.12**. Check if you already have it:
```bash
python --version
```
If it says 3.12.x, you're good. If not, download it from https://www.python.org/downloads/ and during
install **tick "Add Python to PATH"**.

---

## Step 2 — Get the code
```bash
git clone <PASTE-THE-REPO-URL-HERE>
cd "Fraud detection"
```
*(Replace `<PASTE-THE-REPO-URL-HERE>` with the GitHub link of the repo, and use the folder name git
created.)*

---

## Step 3 — Create a clean workspace ("virtual environment")
This keeps the project's packages separate from the rest of your computer.

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\activate
```
**Mac / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```
You'll know it worked when your terminal line starts with `(.venv)`.

---

## Step 4 — Install the packages needed to run it
```bash
pip install -r requirements-serve.txt
```
*(This is the small/fast set — just what's needed to RUN the app. The full `requirements.txt` also
includes training and notebook tools, which you only need if you want to re-train.)*

---

## Step 5 — Start the app
```bash
python flask_app.py
```
You'll see a message like `Running on http://127.0.0.1:5000`.
Now open your web browser and go to: **http://localhost:5000**

Click **"Load genuine sample"** (or **"Load suspicious sample"**), then **"Check Claim"**. 🎉

**To stop the app:** press **Ctrl + C** in the terminal.

---

## That's it!

---

## (Optional) Things you can also do

**Re-train the models yourself** (the repo already has trained ones, but if you want to rebuild them):
```bash
pip install -r requirements.txt            # full toolset
python -m scripts.generate_synthetic_data  # create the sample data
python -m src.train                        # train + save the models (~1 minute)
```

**Open the step-by-step ML notebook in Jupyter:**
```bash
pip install -r requirements.txt
jupyter notebook
# then open notebooks/fraud_severity_project.ipynb and run the cells top to bottom
```

**Read the full project report:** open `reports/report.html` in any web browser.

---

## What are all these technologies? (in plain English)

You don't need to learn these to run the app — here's a one-line "what is it" for each, so it's not
overwhelming:

| Technology | What it does (simple) |
|---|---|
| **Python** | The programming language everything is written in. |
| **Flask** | Turns the model into a web page + button you use in your browser. |
| **scikit-learn** | The main machine-learning toolbox (builds and runs the models). |
| **XGBoost** | A strong type of model we try alongside the others. |
| **SHAP** | Explains *why* the model made a decision (the "top reasons" you see). |
| **pandas / numpy** | Handle the data — tables of numbers. |
| **MLflow** | Keeps a logbook of each training run. *You can ignore it* — it just records things. |
| **pandera / pydantic** | Double-check the data/input is valid so the app doesn't crash on bad input. |
| **Jupyter** | A notebook to explore the machine learning step by step (optional). |
| **Docker** | Packages the app so it runs the same on any computer or server (used for hosting). |

---

## Putting it on GitHub (push)

> **"I committed the code — why isn't it on GitHub yet?"**
> *Committing* saves a snapshot **on your own computer**. *Pushing* uploads it **to GitHub**. Pushing
> needs two things that only **you** can provide: (1) a GitHub repo to upload to, and (2) your GitHub
> login. So these steps must be done by you (or with your sign-in).

1. Go to **https://github.com/new** → give it a name → choose **Private** or **Public** →
   **don't tick** any "Initialize" checkboxes → **Create repository**.
2. Copy the URL it shows (looks like `https://github.com/yourname/your-repo.git`).
3. Run these two commands in the project folder:
   ```bash
   git remote add origin https://github.com/yourname/your-repo.git
   git push -u origin main
   ```
4. A GitHub login window may pop up the first time — sign in to allow the upload.

Done — refresh your GitHub page and the code is there.

---

## Hosting: Netlify vs Render

| Option | Can it host THIS app? | Why |
|---|---|---|
| **Render** | ✅ **Yes** — use this | Render runs full Python/Flask servers (and Docker). Our app needs a running server to load the ML models and answer requests. |
| **Railway** | ✅ Yes | Same as Render — runs a Python/Docker server. |
| **Netlify** | ❌ **No** | Netlify only hosts *static* websites (plain HTML/CSS/JS) and tiny serverless functions. It **cannot run a Python server** with ML models loaded in memory. |

**So: host on Render (free), not Netlify.**

### Deploy on Render (free) — quick steps
1. Push the code to GitHub (steps above).
2. Go to **https://dashboard.render.com** → **New +** → **Blueprint** → connect your GitHub →
   pick this repo → **Apply**. *(Render reads the included `render.yaml` and sets everything up.)*
3. Wait ~5–8 minutes for the build. Render gives you a public link like
   `https://your-app.onrender.com`.
4. Open that link — your app is live on the internet! Put the link in the README.

*(If the repo is Private, Render will simply ask you to authorize access to it — just click allow.)*

---

## Help! Something went wrong

| Problem | Fix |
|---|---|
| `python` not found | Install Python 3.12 from python.org and tick "Add to PATH", then reopen the terminal. |
| `pip install` is slow | That's normal the first time (it downloads packages). Let it finish. |
| Port 5000 is busy | Edit the last line of `flask_app.py` and change `port=5000` to e.g. `port=5050`. |
| "Models not found" error | Run the re-train steps above (`python -m src.train`). |
| Browser shows nothing | Make sure the terminal still shows "Running on ..." and you opened **http://localhost:5000**. |
