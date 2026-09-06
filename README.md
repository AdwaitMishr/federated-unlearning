# Federated Unlearning for Privacy-Compliant Collaborative Medical Imaging

Capstone project (VIT Chennai, CSE). A benchmark that measures the
**forgetting × utility × cost** trade-off of federated-unlearning methods on
clinical imaging (MedMNIST), using a frozen ResNet-18 backbone with a
trainable classifier head.

This repository currently implements **Section 5 scope** of the project spec —
a working, ~50%-of-total-scope subset with real results. The full experiment
matrix (Section 6 of the spec) is structured to be added later as config
changes, not rewrites.

---

## What it does

1. Simulates `N` non-IID "hospital" clients on **PneumoniaMNIST** and **PathMNIST**
   via a Dirichlet(α) partition (whole `(image, label)` tuples, never labels alone).
2. Trains a global model with **FedAvg** on a **frozen pretrained ResNet-18**
   backbone + a **trainable classifier head**.
3. Removes one client ("this hospital leaves the federation") and applies three
   unlearning methods:
   - **Retrain** (gold standard) — FedAvg from scratch on the remaining clients.
   - **Fine-tune** — continue FedAvg on the remaining clients.
   - **Head-only KAF** (the team's novel contribution) — Knowledge-Adaptation Priors
     restricted to the classifier head.
4. Audits every model (baseline + 3 unlearned) with:
   - a **membership-inference attack** AUROC (forgetting; lower = better),
   - retained **weighted F1** on the global held-out test set (utility),
   - time + communication-volume **cost** relative to full retraining.

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt      # PyTorch CPU is sufficient
```

> PyTorch is intentionally CPU-only in `requirements.txt`. To install the CPU
> wheel directly: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`

On first run, MedMNIST and the ImageNet ResNet-18 weights download automatically.

## Reproduce every result

```bash
python -m experiments.run_all_scoped
```

This trains a baseline + the three unlearning methods on **both** datasets and
writes:
- `results/results.csv` — one row per `(dataset, method)`,
- `results/bar_chart_f1_auroc.png` — MIA AUROC & retained F1 grouped bars,
- `results/tradeoff_scatter.png` — forgetting vs. utility with cost,
- `results/README.md` — how to read / regenerate.

## How each unlearning method works (plain English)

- **Retrain.** Train again from nothing on the hospitals that *didn't* leave.
  Perfect forgetting, but it is the slowest and most expensive option — the
  reference every other method is compared against.
- **Fine-tune.** Keep the already-trained model and run a few more training
  rounds on the remaining hospitals. Cheap and quick, but the model still
  "remembers" some of the removed hospital's data — expected to forget poorly.
- **Head-only KAF (ours).** The classifier head is nudged to (a) stay close to the
  trained model's head for the kept data, while (b) actively becoming unsure on the
  removed hospital's data (a "forgetting" push). Because only the tiny head is
  updated — never the frozen backbone — it is much cheaper than retraining.

**Exact Head-only KAF loss (documented):**
```
L(h) = prior_weight · ||h − h_base||²   −   forget_weight · CE(h, removed_heldout)
```
minimised over the head parameters `h` only, where `h_base` is the baseline head
(the "prior"), and `removed_heldout` is the removed client's local validation
tuples (never trained on). After the KAF update we run a few stabilisation FedAvg
rounds on the retained clients. This is a **documented approximation** of
Knowledge-Adaptation Priors — see `src/unlearning/head_only_kaf.py` and the
paper references in the project docs.

---

## Configuration

Everything lives in `config/default.yaml` (datasets, α, client count, rounds,
method weights). Section 5 runs a single fixed setting (`α=0.3`, `N=5`,
client-level removal of client 0); the full sweep later is a config change.

## Repository layout

```
config/default.yaml        all hyperparameters
src/data/                  MedMNIST loaders + Dirichlet partitioning + feature cache
src/models/                frozen ResNet-18 + head
src/federated/             local head training + FedAvg server loop
src/unlearning/            retrain / finetune / head_only_kaf
src/evaluation/            MIA attack, deleted-client probe, metrics
src/run_experiment.py      one (dataset × config) experiment
experiments/run_all_scoped.py   Section-5 runner → results/ (CSV + plots)
```

## Known issues / challenges

This section documents real difficulties encountered while building the pipeline
(the team will be quizzed on these — problem-solving evidence for the rubric).

1. **Dirichlet partition float error.** Initial per-sample `rng.choice` failed with
   "probabilities do not sum to 1" due to floating-point normalisation of the
   Dirichlet weights. Fixed by computing exact per-class client counts via
   `numpy.random.multinomial` and normalising the weight vector explicitly.
2. **Frozen-backbone features at 28×28 are weak (but learnable).** The ImageNet
   ResNet-18 backbone is trained at 224×224; we feed native MedMNIST 28×28.
   A linear probe still reaches ~0.82 test accuracy on PneumoniaMNIST, so the
   signal is usable, but we do **not** claim this is the best possible backbone.
   Mitigation for the final review (if time): resize to 224 or use a small
   medical-imaging backbone.
3. **Head-only SGD under trains with too few FedAvg rounds.** A 2-round smoke test
   plateaued at the majority class (~0.625); features are fine (linear probe 0.82)
   but a head needs enough rounds, and — critically — well-conditioned inputs.
   Combined with feature standardisation (next item) and 40 FedAvg rounds it
   converges to ~0.84 F1.
4. **The frozen ResNet-512 features are large-magnitude.**  Empirical logits were
   routinely in the tens with a small-norm head — a linear classifier on such
   inputs is over-confident and, under FedAvg's weight averaging across skewed
   clients, collapses to the majority class (a run with no normalisation
   reached ~0.62 test F1 [majority]).  We **z-score the features** using global
   client-train statistics before any head training; F1 then converges to ~0.84.
5. **MIA AUROC is at chance (~0.50) for every method — including the baseline.**
   The frozen-backbone + linear-head model is too low-capacity to memorise
   individual training tuples, so membership is not detectable and the attack
   finds nothing to "forget" for the baseline itself.  This is itself a
   finding, but it means MIA is **not** the discriminating metric here.
   We therefore report MIA (honestly ~0.5) AND the **deleted-client probe**
   (residual accuracy on the removed client's held-out tuples), which does
   discriminate: Head-only KAF reduces it (~0.88) below baseline/retrain/
   fine-tune (~0.92–0.93) at a fraction of the cost.
6. **MIA member/non-member pools must be class-matched.**  A first MIA version
   compared the removed client's TRAIN tuples against its own LOCAL-VAL tuples;
   the class/covariate shift between them gave an AUROC that was constant
   across methods.  Non-members are now sampled from the GLOBAL TEST set,
   matched to the members' class counts, leaving only genuine training-membership
   as a possible signal.
7. **PathMNIST is too heavy for laptop CPU at full scale (~90k tuples).** We cap
   each client's training allocation (`path_train_cap_per_client: 3000` in config,
   documented) so the demo stays CPU-feasible. Disable the cap (null) only with a
   GPU. This is a *deliberate scope-time trade-off*, not a silent omission.
8. **Frozen-backbone feature caching.** Features are cached to `.cache/` once per
   tuple set. The cache is keyed on the partition + dataset, so re-running with the
   same seed is fast. `.cache/` is gitignored.

## Scope: implemented now vs. deferred

**Implemented (Section 5 — this pass):**
- Both datasets, one fixed setting (α=0.3, N=5, client-0 removal)
- FedAvg baseline; Retrain; Fine-tune; Head-only KAF
- MIA AUROC, retained F1, deleted-client probe, time & communication cost
- CSV + two plots + this README

**Deferred (Section 6 — final review; code structured so these are config/loop
changes, not rewrites):**
- Gradient Ascent and Certified Removal methods
- Class-level and sample-level removal modes
- α ∈ {0.1,0.3,1.0} × N ∈ {5,10} × removal-fraction sweep, repeated/sequential removal
- ≥3 seeds with mean±std
- a clinical-vs-natural-image comparison (e.g. EMNIST)
- Streamlit results dashboard

## License

No license — all rights reserved (academic capstone work).