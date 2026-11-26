# Data Centre Capital-Project Recovery — Pure LLM vs. AlphaEvolve Demo

Self-contained Google Cloud **AlphaEvolve** demonstration comparing **Pure Gemini LLM scheduling** vs. a **Gen-0 greedy baseline** vs. an **AlphaEvolve-evolved deterministic Python solver** on realistic synthetic hyperscale Data Centre disruptions (80–116 Primavera P6 tasks, Kronos daily trade caps, Free-Issue Material arrival gates, physical work-front concurrency, and Liquidated Damages curves).

> **Privacy-by-Design (Client-Side Split Architecture):** All synthetic Data Centre schedules live strictly under `experiment/data/` (never included in `seed_files`). Only `solver.py` + `schema.py` cross to Google Cloud's AlphaEvolve API, and only numeric scores + diagnostic insights return (`boundary_audit.jsonl` logs `raw_schedule_bytes_egressed: 0` on every iteration).

---

## Repository Structure

```text
datacenter-recovery-alphaevolve-demo/
├── experiment/
│   ├── experiment.yaml         # Official AlphaEvolve declarative config (with Pareto sampling)
│   ├── schema.py               # Immutable domain types (read-only seed context)
│   ├── solver.py               # Gen-0 seed solver with strict col-0 # EVOLVE-BLOCK markers
│   ├── verifier.py             # Shared 5-gate deterministic hard feasibility verifier
│   ├── generator.py            # Reproducible 12-train + 6-held-out DC benchmark generator
│   ├── evaluator.py            # Client-side multi-objective scorer (4D HV, timeout, score=None)
│   ├── evaluator.Dockerfile    # Optional Cluster Toolkit / Cloud Batch worker container
│   └── data/                   # Strictly client-side JSON scenarios (T01-T12 train, H01-H06 heldout)
├── scripts/
│   ├── alphaevolve_client.py   # Zero-dependency REST client for Discovery Engine AlphaEvolve v1alpha
│   ├── run_evolution.py        # Plane A client-side runner (--dry-run | --smoke | --max-programs N)
│   ├── run_llm_baseline.py     # Fair Pure Gemini 2.5 Pro baseline audited by verifier.py
│   └── run_benchmark.py        # 3-way table: Pure LLM vs. Gen-0 Baseline vs. AlphaEvolve Solver
├── artifacts/                  # Outputs written by evolution & benchmark runs
└── tests/                      # Full unittest / pytest verification suite
```

---

## Quickstart (Zero `pip install` Required in Cloud Shell!)

All core scripts use the **Python 3 standard library** (`urllib.request` + `gcloud auth print-access-token` fallback), so you can test directly in Argolis Cloud Shell on **`project-3zcmvkwmrw953`** immediately after cloning.

### 1. Push to Your GitHub (`sharmila-devi-gh/datacenter-recovery-alphaevolve-demo`)
Create an empty **private** repo named `datacenter-recovery-alphaevolve-demo` under `sharmila-devi-gh`, then from your Cloudtop:

```bash
cd ~/datacenter-recovery-alphaevolve-demo
git remote add origin https://github.com/sharmila-devi-gh/datacenter-recovery-alphaevolve-demo.git
git push -u origin main
```

### 2. Verify Locally (Offline — Zero GCP Calls)
```bash
python3 -m unittest discover -s tests -v
python3 scripts/run_evolution.py --dry-run
```

### 3. Run Live AlphaEvolve Smoke Test on `project-3zcmvkwmrw953` (5 Programs)
In Cloud Shell (or any terminal logged into Argolis with `gcloud config set project project-3zcmvkwmrw953`):

```bash
python3 scripts/run_evolution.py \
  --project-id project-3zcmvkwmrw953 \
  --engine-id global-gemini-enterprise \
  --smoke
```

Once the 5-program smoke test confirms end-to-end mutations and score improvement, run a stronger evolution run (30–60 programs):

```bash
python3 scripts/run_evolution.py \
  --project-id project-3zcmvkwmrw953 \
  --engine-id global-gemini-enterprise \
  --max-programs 40 \
  --concurrency 2
```

Outputs written automatically to `artifacts/`:
- `artifacts/evolved_solver.py` — frozen winning Python solver discovered by AlphaEvolve
- `artifacts/best_solver_diff.patch` — readable code diff (`Gen-0 -> Evolved`)
- `artifacts/lineage_curve.json` — generation-by-generation fitness & Pareto curve
- `artifacts/boundary_audit.jsonl` — client-side privacy proof (`raw_schedule_bytes_egressed: 0`)

### 4. Run Fair Pure LLM Baseline (Vertex AI Gemini 2.5 Pro) & 3-Way Comparison
```bash
# Run Pure Gemini directly on Hero held-out scenarios H01 (30d Chiller Slip + 25% Cost Shock)
# and H02 (Water-to-Air Cooling Redesign), audited through verifier.py:
python3 scripts/run_llm_baseline.py \
  --project-id project-3zcmvkwmrw953 \
  --location us-central1 \
  --model gemini-2.5-pro \
  --scenarios H01,H02

# Generate the 3-way comparison table (Pure LLM vs. Gen-0 Baseline vs. AlphaEvolve Solver):
python3 scripts/run_benchmark.py
```

### 5. Launch the CXO Streamlit Executive Demo UI (`app.py`)
Install the UI dependencies inside a virtual environment and launch Streamlit:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py
```

- **In Cloud Shell:** Run `.venv/bin/python -m streamlit run app.py --server.port 8080`, then click **Web Preview (top-right) → Preview on port 8080**.
- **On Your Mac:** Open `http://localhost:8501`.
- **Optional — Deploy to Cloud Run (`sd-proj-382223`):**
  ```bash
  gcloud services enable run.googleapis.com artifactregistry.googleapis.com --project=sd-proj-382223
  gcloud run deploy dc-recovery-alphaevolve-ui \
    --source . \
    --dockerfile Dockerfile.streamlit \
    --region asia-south1 \
    --project sd-proj-382223
  ```

The single-page executive UI automatically loads your live `artifacts/evolved_solver.py` and `artifacts/llm_baseline_result.json` as soon as they exist (and provides a clearly labeled verified preview reference solver so you can rehearse immediately).

---

## Hard Feasibility Gates Audited Identically Across All 3 Approaches (`experiment/verifier.py`)
1. **Completeness & Past-Task Freeze:** Every active P6 task (including cooling redesign rework tasks) scheduled once; completed tasks before `trigger_day` stay locked.
2. **P6 Precedence:** Finish-to-Start ordering (`start_day[task] >= end_day[predecessor]`).
3. **FIM Arrival Gates:** No task starts before its Free-Issue Material (`HV_SWITCHGEAR`, `CHILLER_PLANT_A/B`, `UPS_PDU_RACKS`, `BMS_PANELS`) arrives.
4. **Kronos Daily Trade Headcount Caps:** Daily cumulative crew across all active tasks `<= kronos_daily_cap[trade]` on every day.
5. **Physical Work-Front Concurrency & Recovery Budget Cap:** Concurrent active tasks per physical zone `<= zone_max_concurrent[zone]` and total incremental recovery spend `<= max_recovery_budget_inr_cr`.
