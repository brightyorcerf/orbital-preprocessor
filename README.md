# Orbital Scene Preprocessor

On-board multispectral perception with a deterministic safety envelope around a generative reasoning layer.

[Open the live command centre](https://osp-command-centre.streamlit.app/)

`0.99 mAP@0.5` · `3.69 MB INT8 model` · `62 ms per tile` · `226-byte briefs` · `43,497:1`

`Real TLEs` · `SGP4-propagated contact windows` · `101 contacts of raw imagery vs 1 of briefs`

![OSP command centre](img.jpg)

---

## Why this exists

Start with one number.

A Sentinel-2 scene is about 100 MB. Give a spacecraft a 2 Mbps radio and an 8-minute ground pass and it can move 122.9 MB per contact, enough for exactly one scene, using the entire window.

Now shrink the radio. 32 kbps, and a pass that is no longer assumed.

Take Sentinel-2C's actual element set from CelesTrak, propagate it with SGP4, and compute when it clears a 10° elevation mask over Hyderabad. The answer is two usable contacts a day, the next one 10.2 minutes long. At 32 kbps, derated for pass geometry, that is **1.95 MB**.

The scene doesn't fit. Not "fits slowly", it does not fit, and no amount of patience fixes it, because you'd need a hundred contacts to move a single image and the camera produces the next one long before that. The backlog grows faster than the link drains it, forever.

At that point, on-board inference stops being an optimisation and becomes the only thing that works at all.

| Platform profile | Contact capacity | One 100 MB scene? | Briefs per contact |
| :--- | ---: | :--- | ---: |
| `moi-1a`, 2 Mbps x 8 min *(assumed)* | 122.9 MB | Barely, using the whole pass | 102,400 |
| `skyroot-oam`, 32 kbps x 5 min *(assumed)* | 1.2 MB | No. Not one. | 1,000 |
| `skyroot-oam`, 32 kbps x **10.2 min** *(computed)* | 1.95 MB | No. Not one. | 2,461 |

That third row is the one that matters, because nothing in it was chosen. The duration came out of the propagation; the capacity came out of the duration.

It also corrected the row above it. `contact_minutes_per_orbit = 5.0` was a `DERIVED` guess, and the real geometry says a good Hyderabad pass runs about twice that. The guess was pessimistic by 2x — which is the useful direction to be wrong in, but it was still a guess, and it is now a computation.

OSP's answer is to never downlink the image. Detection runs on-orbit; what comes down is a structured brief: a few hundred bytes saying *what* was found, *where*, and *how confident*.

Run the real corpus through the real window and the comparison is not close:

> **20 tiles as raw imagery:** 20 x 9.83 MB = 197 MB. At this window's 1.95 MB, that is **101 contacts** — about 50 days at 2 usable passes per day.
>
> **The same 20 tiles as briefs:** 15.8 KB. **One** pass, using 0.5% of it. Same 104 detections, 19,863x fewer bytes.

---

## Data flow

```
ON-BOARD (constrained)                    │  GROUND (unconstrained)
──────────────────────                    │  ─────────────────────
6-band tile  (640×640×6, 9.83 MB)         │
  → INT8 detector      62 ms              │
  → per-class NMS                         │
  → pixel → lat/lon                       │
  → protobuf brief     226 B              │
        ↓                                 │
  ┌─ DOWNLINK SCHEDULER ─────────┐        │
  │  committed TLE → SGP4        │        │
  │  → look angles vs station    │        │
  │  → AOS/LOS at elevation mask │        │
  │  → byte budget               │        │
  │  → priority sort → what fits │        │
  └──────────────────────────────┘        │
        ↓  (only what fits)               │
    ~~ contact window ~~ ────────────────┼──→  RAG retrieval (FAISS)
                                          │       → episodic memory (SQLite)
                                          │       → LLM writes the analysis
                                          │              ↓
                                          │       PolicyEngine  ← holds authority
                                          │              ↓
                                          │       reconciled verdict → tasking request
```

Left of the line runs on a compute budget. Right of the line does not.

---

## Four things worth your attention

### 1. The orbit is computed, not drawn

Until recently this project's "orbit" was a cosmetic circle: a perfect 51.6°-inclination ring in `ground/globe.py`, no Earth rotation, and a constant longitude offset applied so the track would pass over the demo scene. Next to it sat `contact_minutes_per_orbit = 5.0` in a config file. Nothing in the repository knew when the spacecraft could actually talk to the ground.

That made the central claim — a brief instead of an image — an assertion rather than a computation.

`orbital/` closes the loop, in layers that each depend only on the one above:

| Module | Does |
| :--- | :--- |
| `tle.py` | Loads a dated, committed CelesTrak snapshot; grades element-set age |
| `frames.py` | TEME → ECEF → WGS-84 geodetic; topocentric look angles |
| `stations.py` | Real ground-station coordinates and elevation masks |
| `propagate.py` | SGP4 → subpoints and elevation profiles |
| `passes.py` | Contact windows, boundaries refined by bisection |
| `downlink.py` | Byte budget → deterministic scheduling decision |

The frame conversions are written out rather than delegated to a library, because that is where satellite geometry code actually goes wrong. Confusing TEME with J2000, or applying GMST with the wrong sign, produces a subpoint on the right continent and a pass of roughly the right length — wrong by tens of kilometres and entirely plausible-looking.

So they are checked against Skyfield, an independent implementation, over 24 hours of propagation. Worst-case disagreement: **38 m in slant range, 0.002° in elevation, sub-metre in altitude.** The residual is exactly the terms `frames.py` documents as deliberately omitted (UT1−UTC, polar motion). Skyfield is a test-only dependency; it is never imported at runtime.

Two assertions pin the ground track to reality rather than to a picture: its latitude bound must equal 180° − inclination (±81.4° for this orbit, where the old fake track topped out at ±51.6°), and successive equator crossings must drift ~25° west, because the Earth turns underneath. A closed synthetic loop fails both.

### 2. The language model is not allowed to be in charge

This is the part I'd most like a reviewer to look at.

There is something genuinely uncomfortable about putting a generative model anywhere near a spacecraft: its failure mode is *confident, fluent, plausible nonsense*. It does not crash. It does not return an error code. It returns a well-formed paragraph that happens to be wrong, and every downstream system accepts it happily.

So OSP structures around that instead of hoping it won't happen:

- `agent/mission_controller.py` computes an alert level deterministically, from the detection payload alone. No model involved. Auditable, reproducible, testable.
- The LLM produces its own independent assessment.
- `_reconcile_alerts()` compares them. When they disagree, the policy engine wins. If the LLM is unavailable, times out, or returns something unparseable, the policy verdict simply stands and the pipeline continues.
- `config/platforms.py` encodes `llm_in_control_loop = False` as a *profile constraint*, a property of the deployment target, not a convention someone can quietly refactor away.

The LLM's job is to explain and contextualise. Never to decide. That distinction is what makes a generative component admissible on a platform that has an assurance requirement at all.

The downlink scheduler is where that stops being a convention and becomes a property of the interface. `DownlinkScheduler.plan()` takes a contact window and a list of briefs. There is no argument, hook, or callback through which a model can reach the decision — not an optional one, not an ignored-by-default one. Priorities come from `score_brief()`, a pure function of a brief's own fields. `DownlinkPlan` is frozen, so the object handed to the analyst for narration cannot be edited by it.

Three tests hold that line, because architecture claims that aren't tested are decoration:

- `test_scheduler_interface_exposes_no_model_hook` inspects the signature and fails if anyone adds an `advisor` or `hint` parameter. That parameter *is* the authority handoff, and this is the review signal.
- `test_plan_is_byte_identical_across_runs` runs the real corpus twice, in opposite input order, and requires identical output.
- `test_narrating_a_plan_cannot_change_it` generates a narrative from a finished plan and asserts the plan is bit-identical afterwards.

Every decision is recorded with the rule that produced it — `fits-in-budget`, `budget-exhausted`, `oversize-brief` — and stamped with a hash of the policy constants. Change a weight and the hash changes, so a plan can never be silently replayed against a policy that didn't produce it.

One result from the real corpus is worth noting because nobody chose it: all five deferrals are `oversize-brief`, not `budget-exhausted`. At a 10-minute pass the *window* isn't the binding constraint at all — the 1024-byte per-payload cap is. That is the kind of thing an assumed 5-minute pass would have hidden.

### 3. Teaching an RGB network to see infrared

YOLOv8n is pretrained on ordinary colour photographs: 3 input channels, 80 output classes. Satellite multispectral data is neither.

The stem swap replaces that first convolution with a 6-channel one (B2/B3/B4/B8/B11/B12, blue through short-wave infrared). The three visible channels inherit their pretrained weights directly. The three new infrared channels are initialised to the mean of the RGB weights rather than to noise, so from the very first training step the network already has working edge and texture detectors on bands it has never seen: infrared reflectance is physically correlated with broadband visible energy, so the RGB mean is a genuinely informative prior, not a hack.

The detection head is rebuilt from 80 COCO classes down to 4 (`ship`, `airplane`, `storage-tank`, `harbor`), carrying over the pretrained bias calibration so the network doesn't start out predicting everything at 50% confidence.

Why bother with infrared at all: B11/B12 short-wave infrared separates man-made hull material from seawater even through light haze, in conditions where the visible bands wash out to uniform grey.

### 4. Every number in this file regenerates from a script

No figure here was typed in by hand. Each one has a command that reproduces it, and the repo distinguishes three kinds of claim:

- `PUBLISHED`: the operator stated it publicly
- `MEASURED`: we measured it on real hardware
- `DERIVED`: an engineering assumption, to be replaced when real specs exist

This matters because an earlier version of this README confidently advertised "85,000:1 compression" and a "<3 MB INT8 model". Both were wrong. The first divided one tile's brief by an entire scene, a 324x coverage mismatch. The second had never been measured. The honest numbers are 43,497:1 and 3.69 MB, and the size target is recorded as missed rather than quietly restated.

---

## Results

### Detection accuracy

80 held-out tiles, none seen during training. Scored through the deployed decision path: same NMS, same class map, same confidence threshold that flight code uses.

| | FP32 checkpoint | INT8 (what ships) |
| :--- | ---: | ---: |
| mAP@0.5 | 0.992 | 0.993 |
| mAP@0.5:0.95 | 0.905 | 0.853 |
| ship *(200 instances)* | 0.985 | 0.990 |
| airplane *(48)* | 1.000 | 1.000 |
| storage-tank *(116)* | 0.983 | 0.982 |
| harbor *(24)* | 1.000 | 1.000 |

Per-class figures are shown because a composite score can hide one dead class behind three healthy ones. Quantization costs essentially nothing at IoU 0.5 and about 5 points at strict IoU, that is, boxes survive, they just get slightly looser.

> Read this honestly: training data is *synthetic*, procedurally generated shapes on flat backgrounds, with clouds and decoy bright specks as hard negatives. It is a much easier problem than real Sentinel-2 imagery. These numbers demonstrate that the training pipeline produces a working detector. They are not an estimate of real-world accuracy.

```bash
python model/evaluate_detector.py --onnx model/artifacts/osp_yolov8n_int8.onnx \
    --images osp_dataset/images/val --labels osp_dataset/labels/val
```

### Quantization

| Metric | FP32 | INT8 | |
| :--- | ---: | ---: | :--- |
| Artifact size | 12.67 MB | 3.69 MB | 3.43x smaller |
| Latency (CPU, 640², sequential) | 110.3 ms | 62.0 ms | 1.78x faster |
| Mean relative divergence | n/a | 3.52 % | max 4.24 % |
| Bitwise determinism | n/a | PASS | identical output across runs |
| `skyroot-oam` 400 ms budget | Missed | Met | 6.5x margin |

Static INT8 post-training quantization (QDQ, per-channel weights), calibrated on real 6-band tiles through the *same* preprocessing path as inference. Static rather than dynamic on purpose: dynamic quantization leaves most convolutions in FP32 and makes latency data-dependent, which breaks the determinism property the assurance story rests on.

Bitwise determinism is the quiet one worth noticing: the same input produces byte-identical output across runs, which is what makes an on-orbit result reproducible on the ground.

```bash
python model/benchmark_quantization.py --platform skyroot-oam
```

### Compression

| Comparison | Ratio | Coverage |
| :--- | ---: | :--- |
| Protobuf brief (226 B) vs raw tile (9.83 MB) | 43,497:1 | same area, the honest per-tile figure |
| Protobuf vs JSON | 2.4x | same content |
| Scene-level, normalised | 1,432:1 | charges the scene all 324 briefs needed to cover it |

Protobuf over JSON buys 2.4x on identical content, largely by sending a single enum byte where JSON spells out `"type": "storage-tank"` in full.

---

## Evaluation

Three suites, because three very different things can be wrong.

`test_pipeline.py`: 13 tests over the engineering path: tensor contracts, geo-projection, protobuf round-trip, compression targets, memory budget, and an accuracy floor on the trained detector. That last one exists because an artifact that exports cleanly, quantizes cleanly and benchmarks cleanly *while detecting nothing* passes every other test in the file.

`test_orbital.py`: 43 tests over the orbital layer. Grouped by what they defend: TLE parsing conventions (an off-by-one on day-of-year shifts every prediction 24 hours while still producing a valid datetime), frame conversions against Skyfield, pass geometry, the scheduling policy, and the authority boundary. Four of them pin the ground track to real orbital mechanics rather than to a plausible drawing.

`ground/eval_suite.py`: 6 axes of LLM faithfulness, scored independently:

| Axis | Catches |
| :--- | :--- |
| `schema_validity` | structurally unusable output |
| `entity_grounding` | omissions, hallucinated detections, class substitutions |
| `coordinate_fidelity` | positions the model invented rather than transcribed |
| `numeric_fidelity` | confidence values that were never downlinked |
| `citation_validity` | fabricated sources, and real sources that were never retrieved |
| `policy_consistency` | disagreement with the policy engine, penalised 2x harder for *under*-escalation |

The composite is the minimum across axes, not the mean. A brief that invents coordinates is not redeemed by having valid JSON.

This replaced a metric that compared `len(anomalies)` to `len(assessments)` and returned a perfect 1.0 on equality, which it happily did for briefs that substituted classes, placed objects 400 km away, fabricated every confidence value, and downgraded a critical scene to nominal. All six are now regression cases.

---

## Tech stack

| Layer | Built with |
| :--- | :--- |
| Detector | PyTorch, Ultralytics YOLOv8n (6-ch stem, 4-class head), custom training loop |
| Runtime | ONNX Runtime, static INT8 PTQ, CPU execution provider |
| Wire format | Protocol Buffers (`osp.proto`) |
| Retrieval | FAISS + sentence-transformers *or* Gemini `text-embedding-004` |
| Reasoning | Google Gemini 2.5 Flash, structured-output mode |
| Memory | SQLite |
| Orbital mechanics | `sgp4` (reference SGP4/SDP4), hand-written frame conversions, Skyfield as a test-only oracle |
| Frontend | Streamlit, Folium (2D), Plotly (3D globe) |
| Deploy | Docker, Python 3.10 |

On the language model: Gemini is called over an API. It is *not* trained, fine-tuned, or hosted here. The only model trained in this repository is the 3.1M-parameter detector.

---

## Run it

```bash
conda create -n osp_dev python=3.10 -y && conda activate osp_dev
pip install -r requirements.txt
```

Train the detector: synthetic dataset, stem surgery, two-phase training, ONNX export, INT8 quantization, scored on both backends.

```bash
python train.py --export          # ~100 min on CPU (26 epochs)
python train.py --quick --export  # ~3 min smoke test of every stage
```

Run inference and produce downlink briefs:

```bash
python inference/engine.py \
  --model model/artifacts/osp_yolov8n_int8.onnx \
  --tiles osp_dataset/images/val \
  --out   data/telemetry_out \
  --platform skyroot-oam
```

Regenerate the committed brief corpus the dashboard serves by default — real INT8 inference over held-out tiles, geolocated on a real propagated ground track:

```bash
python tools/generate_briefs.py        # → data/briefs/ (20 briefs + thumbnails + manifest)
```

Inspect the orbital layer directly:

```bash
python orbital/tle.py                  # element sets in the committed snapshot, with ages
python tools/refresh_tle.py            # fetch a new dated snapshot from CelesTrak
python tools/refresh_tle.py --set-default
```

Launch the command centre:

```bash
streamlit run ground/dashboard.py
```

Verify everything:

```bash
python test_pipeline.py                # 13 tests, engineering path, no API key
python -m pytest test_orbital.py -v    # 43 tests, orbital mechanics + scheduler
pip install skyfield                   # needed for the independent-oracle test
```

`GEMINI_API_KEY` is needed only for the reasoning layer. Everything upstream, perception, quantization, serialization, the policy engine, runs without any key or network access.

> Training defaults to CPU deliberately. On Apple's MPS backend, the identical run that reaches `cls_loss 0.29 / mAP50 0.99` on CPU diverges to `cls_loss 6.0 / mAP50 0.02` with the same seed and the same data. That's a numerical bug, not a speed trade-off.

---

## Tradeoffs and decisions I made

Every choice below had a defensible alternative. What follows is what I picked, what I gave up, and why — including the ones where the honest answer costs me a nicer-looking number.

### Committed TLE snapshot, not a live CelesTrak fetch

**Chose:** a dated element-set file in the repo. The deployed app never touches the network.

**Gave up:** freshness. SGP4 along-track error grows ~1–3 km/day from the element epoch, which shows up as a timing shift in predicted pass times.

**Why:** the public artifact is a Streamlit deployment, and a live fetch makes a third party's uptime into my first impression. It also makes the demo non-reproducible — two readers would see different pass times for the same page.

**Mitigation, because the tradeoff is real:** `TLERecord.staleness()` grades age against thresholds derived from that error growth (`fresh` < 3 d, `usable` < 14 d, `stale` < 45 d), the UI warns when an element set goes stale, and `tools/refresh_tle.py` regenerates the snapshot as a reviewable commit. An old TLE is allowed to produce a number. It is not allowed to produce a number that looks freshly measured.

### Hand-written frame conversions, with Skyfield as a test oracle

**Chose:** write TEME→ECEF→geodetic and the look angles explicitly; use Skyfield only in tests.

**Gave up:** three lines of library code, replaced by ~150 lines I have to be right about.

**Why:** two reasons. Skyfield pulls in JPL ephemeris machinery and wants a downloadable leap-second file — the wrong dependency profile for a project whose argument is *this fits in a constrained environment*. More importantly, the frame conversion is where this class of code goes wrong, and the failure is silent: the answer still looks like an orbit. Writing it out and checking it against an independent implementation is how that bug gets found. Hiding it in a library call is how it ships.

**Result:** 38 m worst-case range agreement over 24 hours, with the residual traceable to documented omissions.

### Modelled: GMST rotation. Not modelled: polar motion, nutation, refraction.

**Gave up:** ~40 m of position accuracy and a second or two of horizon timing.

**Why:** all three sit far below the error already introduced by TLE age. Modelling them would be false precision — arithmetic that looks more careful without being more correct. They're named in `frames.py` so the omission is a decision on record rather than an oversight.

### Constant link rate with a coarse elevation derating

**Chose:** bytes = rate × duration × a factor from peak elevation (0.45 grazing → 0.90 excellent).

**Gave up:** a real link budget. A genuine link varies with slant range — ~2,000 km at the 10° mask versus ~700 km overhead, roughly 9 dB of extra path loss — so a constant rate is *optimistic*, most so on low passes.

**Why:** a proper budget needs antenna gains, noise figures and modulation schemes that this project has no basis to invent. Inventing them would dress an assumption as engineering.

**Mitigation:** the derating is explicit, named (`PassEfficiency`), recorded in the audit trail, and labelled *assumed* in the UI next to the numbers it produces. It is still a model. It is a model whose optimism is visible and adjustable instead of implicit.

### Greedy scheduling, not optimal packing

**Chose:** strict priority order, first fit.

**Gave up:** a few percent of window utilisation. This is a knapsack problem and greedy is not optimal for it.

**Why:** optimal packing improves utilisation by reordering — dropping a higher-priority observation to slot in a smaller lower-priority one. Priority order *is* the mission; utilisation is a diagnostic. Trading the first for the second is a worse system that scores better.

### A hand-written scoring function, not a learned ranker

**Gave up:** ranking quality. A learned model would almost certainly order briefs better.

**Why:** it would also make the answer to "why was this brief dropped?" a matrix multiplication. On a vehicle where every autonomous action must trace to a rule, that is a worse system even when it ranks better. The policy is six named constants readable in one screen — deliberately, because the policy is the artifact a reviewer should argue with.

### 10° elevation mask, not 0°

**Gave up:** roughly half the apparent contact time.

**Why:** dropping the mask to the horizon would nearly double every downlink figure in this project, for free, by counting time when the link cannot close. Real S-band stations work to 5–10°. The mask is stated per station, carried through the pass finder, and shown in the UI, because the honesty of the headline number depends on it.

### Held-out validation tiles for the public corpus

**Gave up:** prettier confidences. Training tiles would score higher.

**Why:** showcasing a detector on data it was fitted to demonstrates nothing.

### Synthetic pixels, real geometry — labelled separately

**Chose:** keep the synthetic imagery, but place every tile at its true subpoint on a propagated Sentinel-2C ground track, and tag each brief with per-component provenance.

**Gave up:** the ability to say "validated on real imagery." I can't say it, so I don't.

**Why:** three different things go into one brief — synthetic pixels, measured detections, real geolocation — and a dashboard that blends them without saying which is which isn't a demo, it's a claim the reader can't check. Each brief carries a `provenance` block; the UI states all three in one line above the map.

**The exact-second version of this:** the imaging campaign was first anchored at 05:44:00Z. That pass is still over inland Maharashtra — a strip labelled "Laccadive Sea" would have been over farmland. Moving the anchor to 05:46:00Z puts the whole strip over water. `test_corpus_is_over_water_in_the_laccadive_sea` now enforces it.

### No synthetic fallback anywhere

**Chose:** if the brief corpus is missing, the dashboard says so and prints the command to build it. If the orbital layer is unavailable, the globe draws no track at all.

**Gave up:** a page that always renders something.

**Why:** this whole piece of work exists because a hand-typed `make_demo_payload()` was the default view of a project that had a trained model sitting unused. A fallback that silently substitutes fiction for measurement is precisely the failure being corrected — rebuilding it as an error path would be rebuilding it.

### Committed thumbnails at 384 px JPEG, not full-resolution PNG

**Gave up:** pixel-level detail; 16 MB became 588 KB.

**Why:** boxes are drawn at full resolution so they land on the exact detected pixels, then the composite is downscaled once. The imagery is context, not evidence — the brief JSON carries bbox coordinates at full precision. Detail no reader can use isn't worth 16 MB of repository weight.

### What I chose *not* to build

Full Sentinel-2 retraining on real imagery. It is the honest next step and it is named in *What this is not*. It is also weeks of work that would lower the headline accuracy numbers, and it would not change the thing this round was about: closing the loop between real orbital mechanics and a real resource decision.

---

## Architecture

A high-level tour is above; the detailed design (module boundaries, the reconciliation state machine, the protobuf schema, and the platform-profile system) lives in [ARCHITECTURE.md](ARCHITECTURE.md).

```
config/     platform profiles + provenance tags
data/       Sentinel-2 preprocessing, synthetic corpus generation
model/      stem swap, training loop, quantization + accuracy benchmarks
inference/  ONNX engine, NMS, geo-projection, protobuf serialization, explainability
rag/        knowledge base + FAISS retrieval
agent/      PolicyEngine and MissionController, the safety envelope
orbital/    TLE ingest, SGP4, frames, ground stations, passes, downlink scheduler
tools/      brief-corpus generation, TLE refresh
ground/     dashboard, 3D globe, episodic memory, LLM analyst, eval suite
data/       preprocessing, synthetic corpus, committed TLE snapshot + brief corpus
```

---

## What this is not

Stated plainly so the scope isn't mistaken for a claim:

- Not validated on real imagery. Training and evaluation are entirely synthetic. Retraining on real Sentinel-2 scenes with real annotations is the next piece of work, not a finished one.
- Not a Skyroot specification. The `skyroot-oam` profile is a `DERIVED` envelope for a launch-vehicle upper-stage compute class, sized an order of magnitude below `moi-1a` so the INT8 and compression work has to genuinely matter. It is not insider knowledge of anyone's hardware and does not claim to be.
- No real-time AIS fusion, no terrestrial vessel-database integration.
- No radiation-hardening certification, no RF regulatory compliance.
- Not a link budget. Downlink capacity is rate x duration with a coarse elevation derating, not a computation from antenna gains and noise figures. See *Tradeoffs*.
- Not a licensed ground station. The Hyderabad site uses Skyroot's corporate coordinates as a planning reference, with a conservative default elevation mask.
- Pass predictions inherit TLE age. The committed snapshot is dated and graded in the UI; a stale element set gives indicative timing, not pointing-grade timing.
- No quantization-aware training: INT8 is post-training only.
- Full atmospheric correction assumed (L2A input).
