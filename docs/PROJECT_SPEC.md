# Project Spec: Federated Unlearning for Privacy-Compliant Collaborative Medical Imaging

**Purpose of this document:** This is a complete technical specification for an AI coding
assistant (Claude Code) with zero prior context on this project. It must contain everything
needed to build, run, and produce results from this codebase without needing to ask the
team clarifying questions. Read this entire document before writing any code.

---

## 1. Project Background (context only — do not skip)

This is a capstone/final-year college project (VIT Chennai, CSE dept, team of 3, guided by a
faculty supervisor). It has already gone through two review presentations (literature review,
problem definition, system design) with **no code written yet**. This is now the third of
four total reviews, and it is graded against a rubric that explicitly requires **"working
modules representing approximately 50% of the approved project scope"** plus real
results, tables, graphs, and analysis — not just a design. A final review will follow, at
which point the full experiment matrix (described in section 6) is expected to be complete.

**Your job in this task is to implement the ~50%-scope subset described in Section 5 below**
— not the full matrix in Section 6. Section 6 is included so you understand the full target
architecture and can build code that generalizes cleanly to it later, but do not try to run the
full sweep now.

### 1.1 The problem being solved

Federated learning (FL) lets multiple hospitals jointly train one shared diagnostic model
without ever moving raw patient scans off their own servers — clients train locally and send
only model weight updates to a server, which aggregates them (FedAvg algorithm).

Privacy law (GDPR Article 17 "right to erasure", HIPAA) requires that if a patient withdraws
consent, a hospital leaves the collaboration, or a data class is invalidated, the trained
model's "memory" of that data must be removed too — not just the raw file. This is the
**machine unlearning** problem, and it is harder in the federated setting because the
deleted data's influence has already been averaged into a global model across many
rounds and many other clients.

Most existing unlearning research evaluates on natural images (CIFAR, EMNIST). This
project argues clinical imaging behaves differently (domain-specific classes, strong
non-IID skew across hospital sites) and builds a benchmark to test multiple unlearning
methods specifically on medical imaging data (MedMNIST), measuring the 3-way tradeoff
between **forgetting quality**, **retained utility**, and **compute/communication cost**.

### 1.2 The team's proposed novel contribution

A lightweight **"Head-only KAF"** variant: since the model architecture already freezes a
pretrained backbone and only trains a small classifier head, unlearning (via
Knowledge-Adaptation Priors, KAF) can be localized to just that head instead of updating
the whole network — trading a small amount of forgetting quality for a large reduction in
compute/communication cost. This is the team's original contribution and **must** be
implemented and shown working — it is the centerpiece of the results.

---

## 2. Tech Stack (use exactly this — matches what was already presented to the panel)

- **Python 3.10+**
- **PyTorch** — model definition, training loops, unlearning methods
- **Flower (`flwr`)** — federated orchestration (simulated multi-client FL). If Flower proves
  too heavy/complex for the timeline, a hand-rolled FedAvg loop (plain PyTorch, simulating
  clients as a Python loop over local `DataLoader`s) is an acceptable fallback — implement
  the FedAvg logic correctly either way. Note this substitution explicitly in the README if used.
- **torchvision** — pretrained ResNet-18 backbone
- **medmnist** (`pip install medmnist`) — dataset loader for PneumoniaMNIST and PathMNIST
- **scikit-learn** — membership-inference attack classifier, AUROC, F1 metrics
- **NumPy / Pandas** — data wrangling, results tables
- **Matplotlib** — plots for the report/slides
- Optional: **Streamlit** — results dashboard (only build this if time remains after core
  pipeline + Section 5 scope is done and stable; not required for this pass)
- **DVC + git** — mentioned in original design for reproducibility; not required for this pass,
  but keep the repo git-friendly (clear structure, no giant binary blobs committed) in case
  they add it later.

All experiments must be **CPU-feasible** (no GPU required) — this is a hard constraint
because the team is using laptops/free Colab, not a cluster. Use 28x28 images (as provided
by MedMNIST), small batch sizes, and keep epoch/round counts modest. Prefer correctness
and clear logging over speed optimization.

---

## 3. Repository Structure to Create

```
federated-unlearning/
├── README.md                     # setup instructions, how to reproduce every result
├── requirements.txt
├── config/
│   └── default.yaml               # all hyperparameters in one place (see Section 7)
├── src/
│   ├── data/
│   │   ├── loaders.py              # MedMNIST loading (Pneumonia + Path)
│   │   └── partition.py            # Dirichlet non-IID client partitioning
│   ├── models/
│   │   └── resnet_head.py          # frozen ResNet-18 backbone + trainable head
│   ├── federated/
│   │   ├── client.py                # local training step (one FL client)
│   │   └── server.py                # FedAvg aggregation loop
│   ├── unlearning/
│   │   ├── retrain.py               # Method 1: full retrain minus target client
│   │   ├── finetune.py              # Method 2: fine-tune on remaining clients
│   │   └── head_only_kaf.py         # Method 3 (novel): KAF restricted to classifier head
│   ├── evaluation/
│   │   ├── mia.py                   # membership-inference attack + AUROC
│   │   ├── deleted_class_probe.py   # class-recognition probe after removal
│   │   └── metrics.py               # F1 (global + per-client), cost (time, comm volume)
│   └── run_experiment.py            # single entrypoint: trains baseline, runs one
│                                     # removal + one unlearning method, evaluates, logs
├── experiments/
│   └── run_all_scoped.py            # runs the exact Section-5 scope end-to-end and
│                                     # writes results/ tables + plots
├── results/                         # output CSVs, PNGs, checkpoints land here (gitignored
│                                     # except a results/README describing the format)
└── notebooks/
    └── sanity_checks.ipynb          # optional, for eyeballing data/partitions/plots
```

Keep functions small, typed where reasonable, and docstringed — a non-technical panel
member may skim the code, and the team needs to be able to explain every function.

---

## 4. Datasets & Non-IID Simulation (needed for both Section 5 and 6)

Use the `medmnist` Python package to download and load:

- **PneumoniaMNIST** — binary classification (Normal / Pneumonia), 5,856 images, 28×28
  grayscale. Splits: train 4,708 / val 524 / test 624 (as provided by the package).
- **PathMNIST** — 9-class tissue classification, 107,180 images, 28×28 RGB. Splits: train
  89,996 / val 10,004 / test 7,180.

**Non-IID client partitioning:** simulate `N` hospital clients by sampling per-class sample
proportions from a `Dirichlet(alpha)` distribution, then assigning each client's local training
set accordingly. Lower `alpha` = more extreme label skew (clients specialize in certain
classes/pathologies); higher `alpha` = closer to IID. Implement this generically for any `N`
and `alpha` — do not hardcode to one value, since the final review will need to sweep this.

Reserve a global held-out test set (not distributed to any client) for evaluating final /
unlearned model utility, and keep track of exactly which samples went to which client (this
is required later for choosing "removal targets").

---

## 5. SCOPE FOR THIS TASK — build and fully run this now

This is the ~50%-of-total-scope subset that must be **working end-to-end with real
numbers**, not stubbed out. Everything below must actually execute successfully and
produce output.

1. **Datasets:** both PneumoniaMNIST and PathMNIST.
2. **Non-IID setting:** ONE fixed configuration — `alpha = 0.3` (moderate skew), `N = 5`
   clients. (Do not sweep alpha or N in this pass — but the code must accept these as
   parameters, not hardcoded constants, so the later sweep is just a config change.)
3. **Federated training:** train ONE global baseline model per dataset using FedAvg —
   frozen pretrained ResNet-18 backbone (ImageNet weights via `torchvision.models.resnet18`,
   `pretrained=True` / `weights=ResNet18_Weights.DEFAULT`) + a small trainable
   classifier head matching the dataset's number of output classes (2 for Pneumonia, 9 for
   Path). Suggested: 20–50 FedAvg rounds, 1–5 local epochs per round per client, Adam or
   SGD optimizer on the head parameters only. Log training/validation loss and accuracy
   per round. Save the trained baseline checkpoint.
4. **Removal target:** ONE removal type — **client-level removal** (i.e., "this whole
   hospital leaves the federation" / withdraws all its data) — remove ONE client (e.g.
   client index 0) from client=5 pool. (Class-level and sample-level removal are NOT
   required this pass, but keep the unlearning method interfaces generic enough to accept
   a `target_indices` or `target_client_id` argument so those modes can be added later
   without a rewrite.)
5. **Unlearning methods — implement all THREE of these, fully working:**
   - **Retrain (gold standard):** re-run the exact same FedAvg procedure from scratch
     using only the remaining `N-1` clients. This is the upper bound on forgetting quality
     and the upper bound on cost.
   - **Fine-tune:** starting from the trained baseline checkpoint, continue FedAvg for a
     few more rounds using only the remaining `N-1` clients (no retraining from scratch).
     Cheap, expected to forget poorly — that contrast IS the point, report it honestly even
     if the numbers look "bad" for this method.
   - **Head-only KAF (the team's novel method):** Knowledge-Adaptation Priors — the
     model is regularized to preserve behavior on the retained clients' data (staying close to
     a prior derived from the baseline / "teacher" model) while pushing to lose distinguishing
     signal for the removed client's data. Crucially, restrict every gradient update in this
     process to **only the classifier head's parameters** — the frozen ResNet-18 backbone
     is never touched. If you are unfamiliar with KAF's exact formulation, implement a
     reasonable, clearly-documented approximation: e.g., a loss combining (a) an L2 or KL
     penalty pulling head weights toward the baseline head's weights (the "prior", to
     preserve retained-client utility) and (b) a gradient-ascent-style term that increases
     loss / reduces confidence on the removed client's held-out samples (to induce
     forgetting), optimized only over head parameters. Document your exact loss formula
     and any simplifying assumptions clearly in code comments and the README, since the
     team must be able to explain and defend this method to the panel.
6. **Metrics — compute all of these for baseline + each of the 3 unlearned models:**
   - **Forgetting:**
     - Membership-inference attack AUROC: train a simple attacker (e.g., logistic regression
       or small MLP from scikit-learn) on model confidence/loss features to distinguish
       "was this sample in the removed client's training data" vs "was this sample never
       in training (held-out)". Report AUROC — closer to 0.5 = better forgetting (attacker
       can't tell), closer to 1.0 = worse forgetting (data is still memorized).
     - Deleted-class probe: N/A for pure client-level removal unless the removed client's
       data happens to concentrate certain classes — if useful, report the model's
       remaining confidence/accuracy on the removed client's local class distribution as a
       secondary signal. Document if skipped and why.
   - **Utility:** weighted F1 score on the global held-out test set, before and after
     unlearning, for the retained-only distribution. Also report per-client F1 on the
     remaining clients if time allows.
   - **Cost:** wall-clock time taken by each unlearning method (measure with
     `time.time()` around the unlearning procedure only, not data loading), and a rough
     communication-volume proxy (e.g., number of FedAvg rounds × number of participating
     clients × parameter count transferred, or just number of rounds run, clearly labeled
     as an approximation) — report both relative to the cost of the Retrain method (i.e.,
     Retrain = 100%, others as a percentage of that).
7. **Outputs required from this pass:**
   - A results CSV/table: rows = {Baseline, Retrain, Fine-tune, Head-only KAF} × {Pneumonia,
     Path}, columns = {MIA AUROC, Retained F1, Unlearning Time (s), Relative Cost %}.
   - At least 2 plots: (a) a grouped bar chart comparing the three unlearning methods on
     MIA AUROC and Retained F1 side by side (per dataset or combined), (b) a
     forgetting-vs-utility scatter/tradeoff plot with the three methods as points, sized or
     annotated by relative cost.
   - Console/log output showing FedAvg training progress and each unlearning method's
     progress, so the team can screenshot/quote real logs during the review if asked.
   - A short `results/README.md` explaining how to interpret the CSV and regenerate the
     plots.
8. **One deliberately-surfaced technical challenge:** while building this, note in the
   README (as a "Known issues / challenges" section) any real difficulty encountered — e.g.
   MIA AUROC being noisy/unstable on a small removed-client sample size, non-IID
   partitioning producing an unusably small client, convergence instability in gradient-ascent
   style KAF loss, etc. — and how it was addressed or mitigated. This is required content for
   the team's presentation (rubric parameter "problem-solving and technical refinement"), so
   do not silently work around issues without logging what happened and why.

**Do NOT build in this pass** (explicitly out of scope, save for later): Gradient Ascent
method, Certified Removal method, the natural-image (clinical-vs-natural) comparison,
class-level or sample-level removal modes, repeated/sequential removal, the full
alpha/N/removal-fraction sweep matrix, and the Streamlit dashboard. It's fine (encouraged)
if the code is structured so these are easy to add later — see Section 6 — but do not spend
time actually running them now.

---

## 6. Full Target Scope (for later / final review — context only, do not implement now)

This is the complete experiment matrix from the original proposal, included so the code
architecture doesn't need to be rewritten later. Design Section 5's code so that each of
these becomes a config change, not a new module, wherever possible:

- All 6 unlearning methods: Retrain, Fine-tune, Gradient Ascent, KAF (full, not head-only),
  Certified Removal, Head-only KAF.
- Both datasets (already covered).
- `alpha ∈ {0.1, 0.3, 1.0}`, `N ∈ {5, 10}`.
- Removal fraction `f ∈ {10%, 25%, 50%}` and removal type ∈ {client, class, sample}.
- Repeated/sequential removal (multiple deletion requests applied one after another).
- ≥ 3 random seeds per configuration, reporting mean ± std.
- A clinical-vs-natural-image comparison set (e.g. EMNIST-style) run through the identical
  pipeline.
- A Streamlit results dashboard.
- Full forgetting × utility × cost "surface" across all axes, and a final method-selection
  recommendation derived from it.

---

## 7. Configuration (put these in `config/default.yaml`, do not hardcode in scripts)

```yaml
dataset: pneumonia   # or "path" — run both, once each, for this pass
alpha: 0.3
num_clients: 5
removal_client_id: 0
fedavg:
  num_rounds: 30
  local_epochs: 2
  clients_per_round: 5      # all clients each round, since N is small
  optimizer: adam
  lr: 0.001
  batch_size: 32
unlearning:
  finetune:
    extra_rounds: 5
  head_only_kaf:
    prior_weight: 1.0        # weight on the "stay close to baseline head" term
    forget_weight: 1.0       # weight on the "forget removed client" term
    epochs: 5
    lr: 0.001
seed: 42
device: cpu
```

Adjust numeric defaults as needed once you've run a few iterations and can see what
actually converges in reasonable time on CPU — but keep every hyperparameter in this file,
not buried in code, since the team will need to tweak and re-run for the presentation.

---

## 8. Definition of Done for This Task

- `pip install -r requirements.txt` followed by running `experiments/run_all_scoped.py`
  (or equivalent single command documented in the README) reproduces every result in
  Section 5 from a clean checkout, end to end, without manual intervention.
- The results CSV and both required plots exist in `results/` after that run.
- README explains: setup, how to run, what each unlearning method does in plain
  English (the team needs to be able to paraphrase this to a panel), the known
  issues/challenges section, and clearly states what's implemented vs. deferred to the
  final review (mirroring Section 5 vs Section 6 of this doc).
- Code is organized per Section 3's structure so extending to the full sweep later is a
  matter of adding config values and loop wrapping, not rewriting core logic.
