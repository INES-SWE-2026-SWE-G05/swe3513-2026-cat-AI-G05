# SWE 3513 Applied AI · CAT 1 (practical) · Group NN

**INES-Ruhengeri · Department of Computer Science · Lecturer: Clement Munyentwari · 7 October 2026**

## The scenario: Kinigi milk collection centre

Every morning farmers around Kinigi, Nyange, Busogo, Gataraga and Shingiro bring
their milk to a collection centre. The collector writes each can in
`data/deliveries.csv`: farmer code, litres, the milk's temperature on arrival and
the hours since milking. Later the lab tests the milk; `rejected` = 1 means the
can was refused. Warm milk, and milk that waited too long, is refused more often.

The centre manager wants:

1. **Clean data** and a **summary per sector**.
2. **A risk score** that warns the collector, before the lab test, that a can may be rejected.
3. **An API** that the collector's phone app (your `swe3409-cat1-gNN` repository) calls.

> All data is synthetic, made up for this CAT.

## Who does what

Every member owns **one AI task and one mobile task**. Agree on member numbers in
the first 5 minutes and write them in your team file.

| Member | AI task (swe3513-cat1) | Mobile task (swe3409-cat1) |
| --- | --- | --- |
| 1 | A1 `data.py`: load and clean the deliveries | M3 `DeliveryList.tsx`: list and totals |
| 2 | A2 `stats.py`: summary by sector, litres by day | M4 `api.ts` + `App.tsx` + `config.ts`: mobile integrator |
| 3 | A3 `model.py`: rejection risk by gradient descent | M2 `DeliveryForm.tsx`: the form |
| 4 | A4 `api.py`: FastAPI service, AI integrator | M1 `logic.ts`: rules, labels, totals |
| 5 | A5 `evaluate.py`: precision, recall, model card | M5 `health.ts` + `StatusBanner.tsx` |

**Group of 4:** there is no Member 5. A5 and M5 stay untouched and nobody is marked on them.

The header comment of each file says which member owns it and what "done" means.

## Tasks in this repository

| Task | File | Tests | Depends on |
| --- | --- | --- | --- |
| A1 | `src/milkcheck/data.py` | `tests/test_a1_data.py` (6) | – |
| A2 | `src/milkcheck/stats.py` | `tests/test_a2_stats.py` (4) | – (own test table) |
| A3 | `src/milkcheck/model.py` | `tests/test_a3_model.py` (5) | – (own test data) |
| A4 | `src/milkcheck/api.py` | `tests/test_a4_api.py` (6) | A1, A2, A3 for /summary and /risk |
| A5 | `src/milkcheck/evaluate.py` | `tests/test_a5_evaluate.py` (4) | – (own test data) |

**When you start your task, delete its `TODO An` lines.** Until then its tests are *skipped*,
so everyone's pull requests stay green. After that, your tests run and must pass.

## Start (once)

```bash
git clone git@github.com:ines-swe-2026/swe3513-cat1-gNN.git
cd swe3513-cat1-gNN
# Windows:  py -3.12 -m venv .venv   then   .\.venv\Scripts\Activate.ps1
# Linux/macOS:  python3 -m venv .venv   then   source .venv/bin/activate
pip install -r requirements.txt
python -m pytest            # 25 skipped: correct, nobody has started
```

## Work cycle (same as Practical 1)

```bash
git switch main && git pull
git switch -c a1-data-aline                 # task - short name - your name
# edit your file, then test ONLY your task:
python -m pytest tests/test_a1_data.py -v
git add src/milkcheck/data.py
git commit -m "A1: drop impossible litres and temperatures"
git push -u origin a1-data-aline
```

On GitHub: **Compare & pull request** → fill in the template → a teammate as **Reviewer** →
after approval, **Merge**. Commit each time a test turns green. Never push straight to `main`.

## Run the API for the phone (Member 4, near the end)

```bash
git switch main && git pull
uvicorn milkcheck.api:app --host 0.0.0.0 --app-dir src
# On the phone's browser: http://<laptop-IP>:8000/health  ->  {"status":"ok"}
```

## How this repository is marked (SWE 3513, out of 100 per student)

Only what is on `main` at the end time counts. See the CAT paper for the full scheme.

| Part | Marks | Where it is read |
| --- | --- | --- |
| Your task's tests pass on `main` | 40 | tests run by the lecturer |
| Your own commits (2 or more) with clear messages | 10 | commit history |
| Your pull request merged | 10 | Pull requests tab |
| You reviewed a teammate's pull request | 5 | reviews |
| `team/<username>.md` filled in | 5 | team folder |
| Group phone demo: the AI part | 20 | live, on the phone |
| You explain or change a line of your own file | 10 | live, at the demo |
