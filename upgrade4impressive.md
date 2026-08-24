# upgrade4impressive.md

**Goal:** make OSP genuinely flagship-worthy as *the* AI project on a top-tier CS resume.
**Timeline:** after the Skyroot email ships (see `upgrade4skyroot.md`). Weeks, not hours.
**Note:** this list and the Skyroot list **barely overlap**. Different audiences, different answers.

---

## Honest assessment: not there yet, for one specific reason

### What is already strong (keep, don't touch)

The **systems half** reads like a senior infrastructure engineer, not a student:

- frame math validated against an independent implementation (Skyfield oracle, 38 m agreement)
- an architecture claim tested **structurally** — `test_scheduler_interface_exposes_no_model_hook`
  fails if anyone adds an `advisor`/`hint` parameter to the scheduler
- per-component provenance labelling (synthetic pixels / measured detections / real geolocation)
- a policy-hash audit trail, so a plan can't be silently replayed against a different policy
- `PUBLISHED` / `MEASURED` / `DERIVED` discipline, and a README that documents its own
  past wrong numbers instead of quietly restating them

Most portfolio projects have nothing like this. It is the differentiator.

### What is hollow — and it's the half people judge

**The model is trained and evaluated entirely on synthetic data the same repo generates.**

Look at a rendered thumbnail: storage tanks are circles, airplanes are plus-signs, ships
are rectangles. `0.99 mAP@0.5` means the network detects literal geometric primitives that
`data/synth_demo.py` drew. That is not a result, it is a tautology — and a sharp reviewer
sees it in about 30 seconds.

Everything downstream then inherits the doubt. The compression ratio, the brief sizes, the
scheduler priorities are all real *given* the detections — but the detections are the weak link.

The README is honest about this (*"Not validated on real imagery"*). Good character. It
**documents** the problem; it doesn't fix it.

> This reverses the earlier "skip Sentinel-2 retraining" advice. That was correct **for
> Skyroot outreach** — Romil doesn't care about boats. It is wrong for a flagship resume
> project, where the perception claim is precisely what's being evaluated.

---

## ITEM 1 — Real data  *(~2–4 days via DOTA, NOT 2 weeks)*

The single blocking issue. Everything else is polish on a hollow core.

> **Correction (2026-08-22):** an earlier version of this file recommended xView3-SAR.
> **That was wrong for this architecture.** Reasoning below. The correct target is DOTA.

### Why NOT xView3-SAR

[xView3-SAR](https://iuu.xview.us/) is superb — 991 Sentinel-1 scenes, 243,018 verified
vessels, a [NeurIPS Datasets & Benchmarks paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/f4d4a021f9051a6c18183b059117e8b5-Paper-Datasets_and_Benchmarks.pdf),
an [AI2 reference implementation](https://github.com/allenai/sar_vessel_detect). It is
also the **wrong sensor modality** for this repo.

xView3 is **SAR**: two polarization channels (VH/VV) of radar backscatter. OSP's detector
is **6-band optical multispectral** (B2/B3/B4/B8/B11/B12). Migrating there means discarding:

- the 6-channel stem swap — one of the three headline technical contributions
- the "SWIR separates man-made hull from seawater through haze" argument
- `data/synthetic_bands.py` and the whole band-derivation story

...in order to rebuild for a different physical sensor. That is not a data migration,
it is a different project. **Do not do this.**

### Why NOT real Sentinel-2 either (the trap)

Sentinel-2 is 10 m GSD. At that resolution:

- a 100 m cargo ship is ~10 px long and ~2 px wide
- a 20 m fishing boat is ~2 px — effectively invisible

This is precisely why serious vessel detection uses SAR or sub-metre optical (Planet,
Maxar). Retraining on real S2 means attempting a genuinely hard research problem with a
small model; realistic mAP lands around **0.2–0.4**. The resulting story is *"I attempted
something hard and did poorly"* — which is worse than the honest-synthetic story you
have now. Small labelled optical S2 ship sets exist
([Roboflow, ~739 images](https://universe.roboflow.com/sentinel2/sentinel-2-ship_detection),
[Kaggle sample](https://www.kaggle.com/datasets/annguynkhc/ship-detection-from-sentinel-2-sample-dataset))
but they do not fix the resolution physics.

### DO THIS: DOTA + the band derivation you already have

OSP's classes are `ship, airplane, storage-tank, harbor`.
[DOTA-v1.0](https://captain-whu.github.io/DOTA/)'s 15 categories include exactly
**Plane, Ship, Storage tank, Harbor**. The synthetic generator was evidently modelled on
DOTA in the first place.

- Real aerial imagery, 2,806 images (800x800 to 4000x4000 px), 188,282 instances
- ~2 GB, [auto-downloads through Ultralytics](https://docs.ultralytics.com/datasets/obb/dota-v2)
- Objects are large and genuinely detectable — no resolution trap
- DOTA-v2.0 exists if more is needed (18 categories, 11,268 images)

**The elegant part:** `data/synthetic_bands.py` already derives 6 bands *from RGB* via
`rgb_to_6band()`. Run it over real DOTA tiles and you get **real pixels, real objects,
real backgrounds, real clutter** — with the 6-channel architecture completely intact.

### Tasks
- [ ] Pull DOTA-v1.0; filter to the 4 classes already in `osp_dataset/dataset.yaml`
- [ ] Convert DOTA's oriented bounding boxes (OBB) to the axis-aligned format the
      current pipeline expects — or adopt OBB properly (ships are rotated; this is a
      genuine accuracy win and Ultralytics supports it natively)
- [ ] Tile the large images to 640x640 to match `INPUT_SIZE`
- [ ] Apply `rgb_to_6band()` → real 6-channel tiles
- [ ] Retrain, re-export INT8, re-run `model/benchmark_quantization.py`
- [ ] Recalibrate INT8 on real tiles (`TileCalibrationReader` — calibration distribution
      must match the new runtime distribution or small-object recall degrades)
- [ ] Regenerate `data/briefs/` from the real model; thumbnails become real aerial imagery
- [ ] Update every downstream figure that assumed the synthetic detector
- [ ] **Write the honest before/after.** Accuracy will drop from 0.99. That is a *feature*.

### Honest labelling after this
> *Real imagery, synthetically extended spectral bands.*

The SWIR/NIR bands remain **derived from RGB, not measured**, so the infrared argument
stays unvalidated — say so explicitly, exactly as the repo already handles provenance.
But "real imagery with derived bands" is enormously more defensible than "synthetic shapes
I drew myself," and it kills the circles-and-plus-signs criticism dead.

### Payoff
`0.99 mAP on shapes I drew` → `0.__ mAP on DOTA`, comparable to a large published
literature. Removes the biggest caveat in *What this is not*. **~2–4 days, not 2 weeks.**

---

## ITEM 2 — Turn the compression claim into an experiment  *(2–3 days, highest novelty per hour)*

**The most under-exploited asset in the repo.** Right now "briefs beat images" is asserted
with one ratio. Make it a measured curve.

- [ ] Plot **detections recovered per byte downlinked** for three strategies:
      (a) raw tiles, (b) JPEG at a sweep of quality levels, (c) semantic briefs
- [ ] Run it over the real contact-window budget from `orbital/downlink.py`
- [ ] Add a rate–distortion framing: what information is *lost* at each operating point
- [ ] Sweep confidence thresholds → briefs are lossy too; show where they break

Nobody has plotted this exact tradeoff. It converts a claim into a result with an x-axis,
and it is the piece with genuine research shape.

---

## ITEM 3 — Real constrained hardware  *(1 day, cheap credibility — but read the risk)*

The repo claims it fits a constrained platform. Currently `DERIVED`.

### Cost of entry
~$100–120 (Pi 5 + PSU + SD card) plus shipping, **if one isn't already on the desk.**
That alone rules it out for any deadline inside a week.

### The risk nobody would warn you about

`config/platforms.py` declares for `skyroot-oam`: `max_inference_latency_ms = 400`,
CPU-only, **2 cores**. Current measured figure is 62 ms — on a Mac.

Published YOLOv8n 640x640 figures on a Pi 5 range from
[~167 ms (INT8 TFLite)](https://github.com/orgs/ultralytics/discussions/8277) to
[over 2,000 ms in some ONNX configurations](https://github.com/ultralytics/ultralytics/issues/21167).
**And this model is 6-channel, so the stem does double the input work.**

Realistic estimate: **300–900 ms at 2 threads. The declared 400 ms budget is a coin flip.**

### Decide before running it
Commit *now* to publishing the number either way.

- **Passes:** `ComputeBudget` / `AssuranceProfile` fields go `DERIVED` → `MEASURED`.
- **Fails:** write *"my own latency budget was too optimistic; here is what would have to
  change"* — for an aerospace audience this is arguably the **stronger** result. It is the
  same self-correction pattern as the `contact_minutes_per_orbit` 2x fix, which is already
  the best hook in `upgrade4skyroot.md`.

Do not run this hoping for a good number. Run it to find out.

### Tasks
- [ ] Run the INT8 engine on Pi 5 / Jetson Orin Nano, **pinned to 2 threads** to honour
      the profile (using all 4 cores would not be measuring what the profile declares)
- [ ] Latency distribution, not just a mean — p50/p95/p99 under sustained load
- [ ] Power draw and thermal throttling behaviour over a long run
- [ ] Compare against `max_inference_latency_ms`; update the profile provenance honestly

**Also high value for the Skyroot email if hardware is on hand — see `upgrade4skyroot.md`.**

---

## ITEM 4 — Fault injection  *(2–3 days full version)*

**Shared with `upgrade4skyroot.md` Task B** — build the minimal version for Sunday, extend here.

- [ ] SEU / bit-flip injection into INT8 weights; degradation curve vs flip count
- [ ] Spectral band dropout (dead sensor)
- [ ] Watchdog timeout → prove `hold_last_known_good_and_flag_ground` fires
- [ ] Corrupted/truncated brief into the scheduler
- [ ] Extend: ECC-style detection, redundant inference, checkpoint/restore
- [ ] Every field of `AssuranceProfile` gets a test that exercises it

Almost no student project does this.

---

## ITEM 5 — CI + one-command reproduction  *(half a day)*

A reviewer will not clone the repo.

- [ ] GitHub Actions running all suites on every push, with a badge
- [ ] `docker run ...` reproducing `data/briefs/` end to end
- [ ] Pin the environment so the numbers are reproducible by a stranger

---

## The uncomfortable meta-point

At the very top end, the strongest student projects are usually **published research** or
**something with real users**. A polished solo portfolio project, however well engineered,
sits below both. Two realistic paths:

### Path A — research shape
Item 2, written as a 4–6 page technical report on arXiv:
*"Semantic compression for bandwidth-constrained Earth-observation autonomy."*
With a real benchmark (Item 1) and a real curve (Item 2), this is a legitimate workshop
paper. Needs Items 1 + 2 done properly.

### Path B — users shape *(downgraded 2026-08-22 — read this before betting on it)*

The original suggestion was to extract `orbital/` as a pip package. **Checked the
landscape; this was oversold.** Satellite propagation is a crowded, mature space:

- **[python-sgp4](https://pypi.org/project/passpredict/0.4.0)** and
  **[Skyfield](https://rhodesmill.org/skyfield/earth-satellites.html)** — Brandon Rhodes;
  the de facto standards
- **[orbit-predictor](https://github.com/satellogic/orbit-predictor)** — maintained by
  **Satellogic**, an actual satellite operator
- **passpredict**, **pyorbital**, and Wikipedia has a whole
  [List of satellite pass predictors](https://en.wikipedia.org/wiki/List_of_satellite_pass_predictors)

`tle.py` / `frames.py` / `propagate.py` / `passes.py` are competent reimplementations of a
solved problem — and they were *validated against Skyfield*, which names the thing a
knowledgeable person would use instead. Shipping it as "another pass predictor" gets ~0
users and risks reading as naive to anyone in the field: *why not just use Skyfield?*

**The one genuinely differentiated piece is `downlink.py`.** "Given contact windows, a link
budget and a queue, decide what to send — deterministically, with an audit trail" is a real
smallsat-operator problem with no obvious open equivalent. That is the only part worth
packaging, and it should be framed as a *scheduler*, not a propagator.

**Realistic ceiling:** a good Show HN might reach 50–200 stars. Genuine adoption — people
depending on it in production, in a narrow aerospace-adjacent niche, from a student with no
track record — is months of community work with a low hit rate.

**Verdict: this is a lottery ticket, not a plan.** Path A has a far better effort-to-payoff
ratio, because a technical report counts on its own merits without needing anyone's adoption.

## Recommended sequencing (post-email)

1. **Items 3 + 5 first** — cheap, immediate, and they harden claims you already make.
2. **Then Items 1 + 2 together** — retrain on DOTA *and* run the bytes-vs-detections
   experiment on the real detector, so one data migration yields both a credible model
   number and a novel result. Item 1 is now ~2–4 days, so this is a single week, not a month.
3. **Then Path A (arXiv write-up).** Path B is a lottery ticket — see the downgrade above.
   If Path B is attempted anyway, package `downlink.py` as a *scheduler*, never `orbital/`
   as another propagator.

---

## Research notes

Generic resume listicles ([DEV](https://dev.to/keerthana_696356/10-resume-ready-ai-projects-for-students-in-2026-with-free-github-ideas-gpo),
[InterviewQuery](https://www.interviewquery.com/p/ai-project-ideas)) say "solve a real
problem, ship a demo, have a live link." **This project is already past that advice** —
which is why the remaining gap is the data honesty problem, not presentation.

---

## STATUS UPDATE (2026-08-23) — plan vs. reality, and where they collide

The DOTA retrain (Item 1) is running on Kaggle right now: non-SMOKE, 8+24 epochs,
`MAX_HOURS = 9.0`, launched via commit-run so it survives disconnects. ETA ~9-10 h from
start. This section reconciles the plan above with what has *actually* landed since it was
written, and flags where the two disagree.

### The doc is stale about its own progress

The "Recommended sequencing" section (above) says do Items 3 + 5 first, *then* 1 + 2
together, *then* Path A. That is not what happened. Commit history shows Items 2 and 4 were
already substantially built *before* this retrain started:

- Item 2 (bytes-vs-detections curve) — `ground/rate_distortion.py`, 526 lines, already
  implements the raw/JPEG-sweep/brief comparison this item asks for, plus a rate-distortion
  plot at `docs/rate_distortion.png`. The confidence-threshold sweep is the one open piece.
- Item 4 (fault injection) — `resilience/faults.py` (303 lines) and
  `resilience/degradation.py` (213 lines) exist, covered by `test_resilience.py` (785
  lines) in CI. Scope against the item's checklist below to see what's actually left.
- Item 5 (CI) — `.github/workflows/tests.yml` exists and runs `test_orbital.py`,
  `test_resilience.py`, `test_pipeline.py` on every push, with skips reported explicitly
  for detector-accuracy tests that need a trained artifact. **Missing:** a badge in
  README, and the `docker run` reproduction has never been verified end-to-end.
- Item 3 (real hardware) — genuinely not started. No Pi/Jetson on the desk yet.
- Item 1 (DOTA) — in flight now, not "not started" as the doc's checklist implies.

**Collision:** treat Items 2, 4, 5 as "mostly done, verify against the checklist" rather
than "do next." Re-running the full multi-day estimate on them would duplicate work
already merged. The only clean next actions are: Item 5's badge + Docker repro proof
(cheap), Item 2's confidence sweep (cheap, needs the new detector), and Item 3 (blocked on
hardware arriving).

### What's safe to do *while the Kaggle run is in flight*

Everything downstream of the new weights is blocked. Everything that prepares for their
arrival is not. In priority order:

- [ ] **Freeze the synthetic "before" numbers now**, into a committed
      `model/artifacts/metrics_synthetic.json` — README prose will be overwritten by the
      refresh below, and the honest before/after this item calls for needs both sides
      committed, not just remembered. Time-sensitive: do this before touching anything else.
- [ ] **Write one refresh command** that chains export → INT8 recalibration on real tiles
      (`TileCalibrationReader`) → `model/benchmark_quantization.py` →
      `tools/generate_briefs.py` → `ground/rate_distortion.py` regeneration → a README-facing
      `metrics.json`. This *is* the back half of Item 1's existing checklist — don't build a
      second version of it, extend the checklist into a script so it runs as one command the
      moment training finishes instead of a manual multi-hour sequence with room for stale
      numbers to survive.
- [ ] **Add a test that fails when README numbers disagree with `metrics.json`.** The
      retrain is exactly the kind of event that silently strands hand-typed figures — this
      repo's whole identity is "every claim is reproducible," so this is closing the
      weakest joint, not scope creep.
- [ ] **Finish Item 5**: README badge, and prove `docker build && docker run` reproduces
      `data/briefs/` end to end. No GPU, no new weights — do this now rather than after.
- [ ] **Latency harness for Item 3**, written and run today against the *current* INT8
      engine under `docker run --cpus 2` with `OMP_NUM_THREADS=2`, to get a same-architecture
      x86 2-thread p50/p95/p99 number now. This does not substitute for the Pi number Item 3
      asks for — it means the Pi run is one command instead of a build-from-scratch when
      hardware shows up. **Order the Pi today if the number matters on a deadline** —
      shipping, not the harness, is the long pole.
- [ ] **Finish Item 2's confidence-threshold sweep** against the *current* (synthetic)
      detector to shake out bugs now, so the rerun against the real detector is a single
      invocation once training completes.

### Resourcing collision: don't add a second training run on this Kaggle session

A natural addition (see below) is an RGB-only YOLOv8n baseline trained alongside the 6-channel
model, to turn "0.__ mAP on DOTA" into a real ablation instead of an isolated number. **Do
not try to fit this into the run that is currently executing.** `MAX_HOURS = 9.0` is sized
for exactly one model on a single Kaggle GPU session per the notebook's own comments; a
second full training run competing for the same session budget risks truncating both. Queue
the RGB baseline as a **second, separate Kaggle commit-run** after this one lands, not a
parallel cell in the same notebook.

### New item — deploy the dashboard (cheap, closes the most commonly cited gap)

Not in the original plan. Hiring-manager-facing research (see below) repeatedly names a
live demo link (Streamlit / HF Spaces) as the single most-scanned-for signal a portfolio
project is missing, ahead of code quality. `ground/dashboard.py` currently only runs
locally.

- [ ] Deploy to Streamlit Community Cloud or an HF Space
- [ ] **Check before deploying:** `ground/llm_analyst.py` calls an LLM API — a public
      deployment needs its own scoped key with a spend cap, not a copy of a personal one.
      `ground/osp_memory.db` and `data/briefs/` ship real (if synthetic-derived) detection
      data — confirm nothing in there should stay private before it's world-readable.
- [ ] This has no dependency on the DOTA retrain and can happen in parallel with anything
      above.

### Path A timing — the ICLR window already closed

Path A (arXiv / workshop write-up of Item 2's curve) is confirmed as the better bet over
Path B (packaging `orbital/`) — no change there. But the plan didn't have deadlines, and
one relevant window already passed:

- **ML4RS @ ICLR 2026** — deadline was **2026-02-06**, already gone. Next cycle
  (ML4RS @ ICLR 2027) will open on a similar late-Jan/early-Feb 2027 cadence.
- **EarthVision @ CVPR** — 2026 edition's CFP is live at
  `cmt3.research.microsoft.com/EarthVision2026`; deadline not yet published as of this
  writing, historically ~March. Worth checking directly rather than assuming a date.
- **NeurIPS 2026 workshops** — suggested submission date **2026-08-29**, six days from
  today. Not realistic for this write-up given Item 1 hasn't finished training yet.

**Practical target: EarthVision or ML4RS's next cycle (~early-to-mid 2027), not a September
2026 deadline.** That's a feature, not a problem — it means Items 1 and 2 can be done
properly rather than rushed to fit a submission window that was never really in reach.

### New item — turn the DOTA number into an ablation, not an isolated figure

Not in the original plan. "0.__ mAP on DOTA" only means something next to (a) published
DOTA baselines and (b) an RGB-only YOLOv8n trained on the same tiles. The README already
proves, by arithmetic, that killing B11+B12 costs 0.000 mAP on the synthetic corpus (*What
this is not*, current README) — the DOTA run is the chance to show whether that holds on
real imagery too, which is a stronger and more specific claim than "we retrained on real
data."

- [ ] Train an RGB-only YOLOv8n on the same DOTA tiles (**separate Kaggle run** — see the
      resourcing collision above, do not fold into the run in progress)
- [ ] Report both mAP figures side by side, plus the existing band-ablation arithmetic,
      as one table
- [ ] If the 6-channel model doesn't beat RGB-only on real imagery, say so as plainly as
      the README already says it about the synthetic corpus — a second honest negative
      result is a stronger artifact than forcing a positive one

### Sources consulted for the hiring-signal claims above

- https://machinelearningmastery.com/7-machine-learning-projects-to-land-your-dream-job-in-2026/
- https://letsdatascience.com/blog/the-ml-portfolio-that-actually-gets-you-hired-in-2026
- https://www.interviewnode.com/post/ml-engineer-portfolio-projects-that-will-get-you-hired-in-2025
- https://ml-for-rs.github.io/iclr2026/
- https://www.grss-ieee.org/events/earthvision-2026/
- https://blog.neurips.cc/2026/08/10/announcing-the-neurips-2026-workshops/
