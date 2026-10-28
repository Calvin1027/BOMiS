# Categorical-only baseline generators

This implementation emits synthetic minority feature rows from a retained binary
training table. All feature columns are categorical. It contains no support
estimator and does not read validation/test rows during fitting or sampling.
The frozen data remain in `prepared_data/categorical_imbalance_v1`.

## Algorithms and provenance

| CLI name | Implementation | Minority sampling |
|---|---|---|
| `smote` | imbalanced-learn SMOTE; one-hot encode, then column-wise argmax | Direct resampling; explicitly an OHE/projection variant |
| `smoten` | Official imbalanced-learn SMOTEN, nominal VDM neighbors | Direct resampling |
| `smote_enc` | Generalized SMOTE-ENC port reused from Baseline_exp_v4 | Label-association encoding and minority neighbor modes |
| `ctgan` | V4 adapter using the official CTGAN package | Native target condition, followed by target/feature checks |
| `tvae` | V4 adapter using the official TVAE package | Joint generation and bounded target rejection |
| `tabsyn` | V4 joint-generation policy; official TabSyn VAE, EDM, decoder | Joint generation and bounded target rejection |
| `tabnat` | Official TabNAT masked transformer and categorical loss | Official joint sampler and bounded target rejection |
| `arf` | Official Python ARF, FORDE and FORGE; all columns are factors | Joint generation and bounded target rejection |
| `stasy` | Official NCSN++, VE-SDE, SPL, likelihood fine-tuning and ODE sampler | Joint generation and bounded target rejection |
| `tabdiff` | Official TabDiff denoiser and learned categorical noise schedule | Joint masked diffusion and bounded target rejection |

Exact upstream commits and copied V4 modules are recorded in
`configs/sources.lock.json`. Upstream checkouts, including their licenses, live in
`third_party/`. `scripts/fetch_sources.py` restores pinned commits without resetting
an existing checkout. No mixed-type benchmark source or result was modified.

ARF is a tree-based generator, not a neural model. SMOTE-ENC is different from
SMOTENC; the latter does not support categorical-only inputs and is not used here.

## Categorical-only compatibility

- `numerical_columns=()` is mandatory. Numerical feature declarations fail early.
- Train-only vocabularies preserve literal strings (`?`, `nan`, leading zeros),
  column order and constant columns. No dummy numerical variable is inserted.
- Internal one-hot encodings represent original nominal variables; they do not
  discretize continuous source measurements.
- TabSyn receives numerical tensors of shape `(batch, 0)`. The absent numerical
  reconstruction loss is zero; categorical CE and KL retain their native meaning.
  Its VAE scheduler uses training CE, avoiding upstream test-file access. VAE/EDM
  architecture, latent scaling, KL schedule, and argmax feature decoding are retained.
- TabNAT uses `x_num=None`. TabDiff uses its native zero-numerical branch.
- STaSy uses the official self-paced weights, EMA and likelihood-based fine-tune
  selection on retained training rows only. Final prescribed-stage weights are
  selected instead of the original demo's held-out downstream-score checkpoint.
  Fine-tuning can be disabled explicitly; this is recorded as a distinct recipe.
  Its discrete noise table is moved to the model device before GPU denoising;
  upstream otherwise indexes a CPU tensor with CUDA indices.
  The same registered model runs directly on the requested device, avoiding the
  upstream factory's unconditional DataParallel routing. Final-stage EMA is
  installed after fine-tuning before sampling, as in native evaluation.
- TabDiff uses the native AdamW and updates EMA once per epoch, including the
  learned categorical schedule, matching the pinned Trainer.
- SMOTE-ENC removes self by row index when neighbor distances tie. Duplicate rows
  remain distinct observations. The previous `[:, 1:]` shortcut was insufficient
  when the point itself was not returned first.
- Bounded rejection counts every decoded row and fails with `CAPACITY_FAILURE`
  when it cannot obtain the requested number. No ROS backfill, majority relabeling,
  generated-row deduplication, or hidden minority-only refit is performed.

Plain SMOTE followed by one-hot argmax almost surely returns one of the two parent
rows: its single interpolation coefficient is shared across all columns. Keep its
`one_hot_argmax` label and `projection_degeneracy` metadata when reporting it.
It is not an independent new-combination generator.

TVAE has two **explicit** decoder settings. `official_argmax` is the reference
recipe. The reused V4 `probabilistic_softmax` option samples only the target span
from its softmax, keeping official feature decoding; it must be reported as a
modified TVAE sampling variant. The short engineering smoke recipe uses this
option to exercise minority-output plumbing even before meaningful training.
It is not silently substituted into the reference recipe.

## Running

Use Python 3.10+ with the dependencies in `pyproject.toml`; on this cluster Python
and model execution must run through Slurm. Upstream models require their pinned
source checkouts. An editable installation or `PYTHONPATH=$PWD/src` exposes the CLI.

```bash
python -m categorical_baselines.cli \
  --model smoten \
  --train prepared_data/categorical_imbalance_v1/voting/split-20261001/train.csv.gz \
  --config configs/models.reference.json \
  --seed 20261001 --rows 169 --device cpu \
  --output runs/voting-smoten-example
```

The output directory must not already exist. Each run records input/config hashes,
separate fitting/sampling seeds, package versions, fitting artifacts, generated
rows and complete rejection accounting. All successful outputs are validated
against training domains. Failures remain visible in `status.json`.

`models.reference.json` supplies executable reference recipes, not a frozen paper
benchmark. Its long training schedules are not used in the implementation checks.
`models.smoke.json` deliberately uses small architectures and short schedules.
Successful smoke generation establishes execution and schema compatibility, not
convergence, generative quality, or success on every imbalanced split.

## Verification

`tests/test_contracts.py` checks encoding round trips, forbidden numerical inputs,
SMOTE projection behavior, SMOTE-ENC distance ties, deterministic resampling,
input preservation and rejection-budget accounting.

`scripts/validate_models.py` runs each algorithm in a separate process, then
independently reloads and checks the generated CSV. Both rounds check exact
encoding/decoding for all eight frozen training schemas. Round 1 trains on a small
artificial categorical fixture. Round 2 uses small balanced subsets drawn only
from the retained Voting and Splice training rows. These are engineering fixtures,
not replacements for the frozen imbalanced training sets.

The cluster wrapper `scripts/validate.sh UNIQUE_TAG 1` (or `2`) transports source
and data to compute-node scratch, runs compilation, Ruff, pytest and GPU smoke
cells, and returns a verification archive under `validation/`. It uses an isolated
dependency overlay: the existing V4 generation packages are read-only inputs,
and the extra TabNAT/STaSy/testing packages are unpacked from
`/mnt/data1/calvinkim1027/categorical_baselines_runtime_20260929/extras.tar.gz`
into each job's scratch directory. This avoids thousands of small dependency
reads/writes over the shared filesystem.
Reports preserve unsuccessful attempts as well as successful rounds.

For a short check on the complete retained imbalanced Voting training table:

```bash
bash scripts/validate.sh UNIQUE_TAG 2 retained voting
```

Use `splice` or `both` in the final argument to select the other retained inputs.
The Python driver also accepts `--models` to check selected adapters.

### Completed checks (2026-09-29)

| Artifact under `validation/` | Scope | Result | Slurm job |
|---|---|---|---|
| `round1_verified/report.json` | Small categorical fixture, 10 adapters | 10/10 pass | 250383 |
| `round2_verified/report.json` | Balanced retained-train subsets of Voting and Splice | 20/20 pass | 250384 |
| `retained_voting_check/report.json` | Full retained Voting train: 187 majority, 18 minority, 16 categorical features | 10/10 pass | 250408 |
| `tvae_official_reference/status.json` | Reference TVAE, official argmax, 300 epochs, same full Voting train | Fit completed; minority sampling capacity failure | 250414 |

**The 40 passing smoke cells include `TVAE-probabilistic-target`, not reference
TVAE.** Each successful cell emitted 16 rows and passed an independent CSV reload,
column/domain/count check. Compilation, Ruff and all 9 unit tests passed in each
of the first three jobs. Each also checked exact categorical codec round trips
on all eight full frozen training schemas. These schema checks do not establish
model-training success on all eight datasets.

Reference TVAE completed all 300 requested epochs. Of 3,200 decoded joint rows,
3,193 had target 0 and only 7 had target 1; all seven minority rows were valid.
The requested 16 rows could not be delivered within the fixed budget, so the run
records `CAPACITY_FAILURE` and produces no completed synthetic dataset. This is
an observed minority-sampling limitation for this seed/recipe, not a categorical
type error. The reference config retains official argmax; the passing smoke
variant is not a replacement benchmark result.

The adapter source SHA-256 for the initial implementation checks above was
`2dc6f3a0188a29313e1b5394704a0a0c074d4ba8bf9ad8bd599c3ae30d0827b4`,
matching every run above. Original V4 source hashes were rechecked unchanged.
Returned archives have SHA-256 sidecars; per-cell metadata, checkpoints, losses,
sampling accounting, logs and unit-test XML are retained. The earlier
`round1_initial` failure is also preserved: its STaSy GPU noise-table indexing bug
was fixed before the verified runs.

No full eight-dataset performance benchmark or generative-quality evaluation has
been run by this implementation task.

## Three-pass code review

The subsequent review and corrections are documented in
[`audits/code_review_20260929/README.md`](audits/code_review_20260929/README.md).
The reviewed snapshot adds regression coverage and a native-decoder probe:

```bash
bash scripts/validate.sh UNIQUE_GPU_TAG 2 fixture both review
bash scripts/validate.sh UNIQUE_CPU_TAG 2 fixture both review-cpu
```

The GPU probe trains all ten adapters on small training-only subsets of all eight
datasets and a constant-feature/single-row-final-batch fixture. The CPU probe runs
the latter fixture on a GPU-equipped node. Joint generation requests 1 and 33
rows and verifies exact seeded replay. **TVAE uses official argmax in this probe.**
It records categorical compatibility and minority-sampling capacity separately;
`CAPACITY_FAILURE` is not a successful minority dataset.

The final reviewed snapshot passed all 23 unit tests and 100/100 categorical
compatibility cells (90 GPU, 10 CPU). Minority sampling delivered the requested
16 rows in 93 cells; the remaining 7 TVAE/STaSy cells recorded capacity failures
under the short recipes. Independent verification reloaded and checked 233
generated CSVs. See the review report for the exact cells, seeds, hashes, and
limits of these checks.
