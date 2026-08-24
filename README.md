# Orbital Scene Preprocessor

A spacecraft decides what to downlink on its next contact, under real orbital and link constraints, by deterministic rule. A language model helps explain the result and is architecturally unable to change it.

[Open the live command centre](https://osp-command-centre.streamlit.app/)

[![tests](https://github.com/brightyorcerf/orbital-preprocessor/actions/workflows/tests.yml/badge.svg)](https://github.com/brightyorcerf/orbital-preprocessor/actions/workflows/tests.yml)

`86% of a corpus's objects: 0.89 MB as briefs, 2.37 GB as raw, unreachable by JPEG` · `one contact buys 1,343 briefs or 0.5 raw tiles`

`SGP4-propagated contact windows` · `10.2 min pass, computed not assumed` · `frames validated to 38 m against Skyfield`

`LLM in control loop: False, enforced by the interface` · `every declared fallback has a test that fires it`

`3.69 MB INT8 detector` · `169 ms per tile end to end on 2 cores` · `155-byte briefs` · `0.880 mAP@0.5 on 3,677 real DOTA tiles`

![OSP command centre](docs/img.jpg)

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

## Five things worth your attention

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

### 3. The compression claim is a curve, not a ratio

The argument for this whole architecture is that a brief beats an image. Until recently that was carried by a single number — 43,497:1, now 63,279:1 on real imagery — and a single number cannot carry it.

The brief corpus shows why. Its per-scene ratios range from 19,582:1 to 31,108:1, and the largest belongs to `P0019_01920_00480`, a brief containing **zero detections**. An empty brief is nearly free, so the headline figure was partly measuring how empty the scenes happened to be. That is a property of the dataset, not of the method. The effect survived the move to real imagery: the widest and narrowest ratios in the DOTA brief corpus are still separated by an empty scene at one end and a two-object scene at the other.

A ratio also answers the wrong question. An operator does not ask how small a brief is. They ask: *given the bytes this pass affords, how much of what is down there will I know about?*

`ground/rate_distortion.py` fixes a byte budget and spends it three ways — raw lossless tiles, JPEG across a quality sweep, and briefs across a confidence sweep — then counts what the ground ends up knowing about the **whole** corpus. Objects on tiles that never fit the budget count as missed.

![bytes versus detections](docs/rate_distortion.png)

That denominator is the entire experiment. Score only the tiles that were delivered and every strategy trends to 1.0, which is exactly the flattering non-result this replaces.

Priced over **1,000 held-out DOTA tiles, 9,472 labelled objects**, sampled at even stride across the val split so all four classes appear in corpus proportion. Contact budget is a `skyroot-oam` pass: 32.0 kbps x 5.0 min = 1,200,000 B.

| Strategy | Bytes/tile | Recall | Precision | Tiles per contact |
| :--- | ---: | ---: | ---: | ---: |
| raw, lossless | 2,366,944 | 0.862 | 0.920 | 0.5 |
| JPEG q30 | 26,079 | 0.814 | 0.925 | 46 |
| JPEG q2 | 8,291 | 0.507 | 0.846 | 145 |
| brief @ conf 0.20 | 951 | 0.893 | 0.873 | 1,262 |
| brief @ conf 0.35 | 894 | 0.862 | 0.920 | 1,343 |
| brief @ conf 0.65 | 755 | 0.707 | 0.974 | 1,590 |
| brief @ conf 0.80 | 273 | **0.000** | **0.000** | 4,396 |

**The brief at conf 0.35 ties raw lossless exactly** — 0.862 recall, 0.920 precision — for **1/2,648th** of the bytes. Not approximately: the same numbers, because the brief *is* the raw tile's detection result at the deployed threshold, and the ground station runs the same detector either way. Reaching that recall costs **2.37 GB** as raw tiles or **0.89 MB** as briefs. JPEG never reaches it at any quality setting: it peaks at 0.828 recall at q75, for 53.3 MB.

Three findings matter more than the headline gap.

**The detector has a confidence ceiling, and past it the brief goes silent.** At conf 0.80 the sweep emits **zero detections across all 9,472 objects** — not a degraded brief, an empty one, still costing 273 B/tile in envelope overhead. The maximum confidence this model produces anywhere is 0.683. The old synthetic detector still returned boxes at 0.90; this one does not, so any operator threshold above ~0.68 silently downlinks nothing. That is a hard operating-envelope constraint and the single most important number in this table.

**Briefs trade recall against precision, and neither is free.** Precision is no longer the flat 1.000 the synthetic corpus reported: it runs 0.754 at conf 0.05 up to 0.974 at conf 0.65, while recall falls 0.921 to 0.707 across the same span. The knee is at conf 0.35, which is where the deployed threshold already sits. The synthetic finding that briefs were lossless in precision was an artefact of a corpus where every object was a distinct geometric primitive.

> **Retracted.** An earlier version of this section reported that heavy JPEG made the detector *hallucinate* — precision collapsing to 0.279 at q2 while recall fell to 0.562, so artefacts were manufacturing false objects. **That does not reproduce on real imagery.** At q2 on DOTA, recall falls to 0.507 but precision holds at 0.846, close to the raw baseline's 0.920 rather than collapsing. Heavy JPEG on real scenes hides objects; it does not invent them. The original result was a property of synthetic tiles, where compression artefacts on flat backgrounds resembled the drawn primitives the detector was trained on. Removed rather than restated.

> **Sampling note.** `--limit` on this script and on `model/evaluate_detector.py` used to take the *first* N tiles. DOTA tile names sort by source image, so a prefix is a contiguous run of a few scenes: the first 1,000 tiles of this split hold 16,433 ships and **zero** storage-tanks. An earlier draft of this table was built that way and was wrong. Both tools now sample at even stride, which reproduces the corpus class balance to within one point per class.

The comparison is set up to be unkind to OSP in three specific ways. Raw is priced as lossless PNG over six uint16 planes, not the 9.83 MB float32 array actually held in memory. Briefs are priced as minified JSON, when the protobuf encoding they really ship in is 2.4x smaller. And ground-side detection uses the same detector at the same threshold as onboard, so the pixel strategies are never handicapped by a weaker analyst.

JPEG is the fair lossy baseline here for a specific reason: the six bands are derived from RGB by a fixed linear map, so a tile's information content *is* its RGB. Compressing the RGB and re-deriving loses what the codec loses and nothing more. That would not hold for a sensor that measured its infrared bands independently — see *Spectral bands* under Results.

What the curve cannot show is worth stating alongside it. Pixels can be re-analysed later, with a better model, for a question nobody has asked yet. A brief cannot. The plot measures one axis of value and the architecture trades away another.

```bash
python ground/rate_distortion.py --tiles val/images --labels val/labels --limit 1000
```

### 4. The declared safety behaviours are executed, not just declared

`config/platforms.py` says, for the `skyroot-oam` profile:

```python
watchdog_timeout_s        = 5.0
max_inference_latency_ms  = 400.0
fallback_on_model_failure = "hold_last_known_good_and_flag_ground"
```

For most of this project's life, none of those were reachable by any code path. That is a worse state than having no declaration at all: it reads as a safety property and behaves as a comment. `resilience/` closes it.

`inference/engine.py` no longer raises. `run_tile` is guarded: any exception out of the perception path, and any pass that overruns the watchdog, is converted into the profile's declared fallback brief, flagged `degraded` so the ground cannot mistake it for a fresh observation. Each declared fallback string resolves to a real handler, and **an engine refuses to start against a profile whose declared fallback has no implementation.** A profile cannot promise a behaviour the code cannot perform.

`test_resilience.py` then forces each failure on purpose:

| Fault | What the system does | Test |
| :--- | :--- | :--- |
| Model crash, execution provider fault | Declared fallback fires, brief flagged `degraded` | `test_a_model_crash_produces_the_declared_fallback` |
| Perception overruns the watchdog | Same fallback, fault recorded as `WatchdogExpiry` | `test_a_stall_trips_the_watchdog_and_fires_the_fallback` |
| Over the latency budget but returning | Reported; the brief still stands | `test_a_latency_budget_breach_is_reported` |
| Failure on the first tile, nothing to hold | Degrades to an empty flagged brief, invents nothing | `test_hold_with_no_history_degrades_further_rather_than_inventing` |
| Truncated or malformed brief | Quarantined with a reason, contact still planned | `test_structurally_destructive_corruption_is_quarantined` |
| Bit flips in INT8 weights | **Nothing. Nothing at all.** | `test_an_upset_model_still_loads_and_runs` |

`test_every_assurance_field_is_exercised` is the one that keeps this honest: it fails if a field is added to `AssuranceProfile` without a test that makes it happen. The same idea as `test_scheduler_interface_exposes_no_model_hook`, pointed at failure behaviour instead of at the authority boundary.

Two results are worth more than the machinery.

**Bit flips are invisible.** A single-event upset is the characteristic failure of flight compute, and it lands in INT8 weights as silent numerical corruption. Flip a quarter of a million bits and the graph still loads, every tensor still has the right shape, inference still returns, and nothing anywhere reports a problem. Accuracy holds to about 0.1% of weight memory and then collapses, and **as it collapses the model emits more detections, not fewer**: 119 at baseline, 6,573 at the far end. The failure mode is not silence, it is confident nonsense. This is the one fault in the table the declared fallback cannot catch, because there is no error to catch. It is the same argument this repo makes about language models, and it turns out to apply to the detector too.

**A single flipped byte in a brief is often undetectable.** About half the time it lands somewhere that still parses and still type-checks, and ingest returns a brief that is well-formed and wrong. Structural validation cannot fix this; an integrity check on the wire would, and OSP does not have one. `test_a_single_flipped_byte_can_survive_ingest_undetected` pins the gap rather than letting the quarantine tests imply a completeness they do not deliver.

What ingest does guarantee is narrower and worth stating precisely: it never raises, and it never repairs. A truncated brief whose anomaly list did not survive is rejected, not coerced to "zero detections", because that is not a missing observation, it is a false one, and the scheduler would spend real bytes downlinking it.

```bash
python resilience/degradation.py          # regenerates the curve
python -m pytest tests/test_resilience.py -v
```

### 5. Every number in this file regenerates from a script

No figure here was typed in by hand. Each one has a command that reproduces it, and the repo distinguishes three kinds of claim:

- `PUBLISHED`: the operator stated it publicly
- `MEASURED`: we measured it on real hardware
- `DERIVED`: an engineering assumption, to be replaced when real specs exist

This matters because an earlier version of this README confidently advertised "85,000:1 compression" and a "<3 MB INT8 model". Both were wrong. The first divided one tile's brief by an entire scene, a 324x coverage mismatch. The second had never been measured. The honest numbers are 63,279:1 on the current DOTA brief corpus (43,497:1 on the synthetic one it replaced) and 3.69 MB, and the size target is recorded as missed rather than quietly restated.

---

## Results

### Detection accuracy

**These numbers are from real aerial imagery.** An earlier version of this section scored 0.992 against tiles `data/synth_demo.py` drew — storage tanks as circles, airplanes as plus-signs, ships as rectangles — and said plainly that the number was close to meaningless, that `data/dota_prep.py` and `tools/kaggle_train_dota.ipynb` existed to replace it, and that the numbers would drop when they did. They did drop. That was the point.

**3,677 held-out tiles** from DOTA-v1.0, 34,918 labelled instances, none seen during training. Scored through the deployed decision path: same NMS, same class map, same confidence threshold that flight code uses.

| | FP32 checkpoint | INT8 (what ships) |
| :--- | ---: | ---: |
| **mAP@0.5** | **0.889** | **0.880** |
| **mAP@0.5:0.95** | **0.576** | **0.544** |
| ship *(19,651 instances)* | 0.960 | 0.952 |
| airplane *(5,464)* | 0.944 | 0.930 |
| harbor *(4,626)* | 0.845 | 0.844 |
| storage-tank *(5,177)* | 0.807 | 0.794 |

**Before and after, on the same code path:**

| | synthetic shapes | real DOTA aerial |
| :--- | ---: | ---: |
| mAP@0.5 | 0.993 | 0.880 |
| mAP@0.5:0.95 | 0.853 | 0.544 |
| held-out tiles | 80 | 3,677 |
| labelled instances | 388 | 34,918 |

The 11-point drop at IoU 0.5 and 31-point drop at strict IoU is what replacing drawn primitives with photographed scenes costs. The synthetic score was not wrong, it was measuring the wrong thing.

Per-class figures are shown because a composite score can hide one dead class behind three healthy ones, and here it would: **storage-tank recall is 0.653** at INT8 against precision of 0.936. The model is not confusing storage tanks with something else, it is failing to find them — 3,610 predictions against 5,177 ground-truth instances. Circular tanks in the synthetic corpus were trivially separable; real ones sit in refinery clutter at varying scale, and the axis-aligned box conversion (mean 1.73x area inflation, see `prep_manifest.json`) hurts tightly-packed tank farms more than it hurts isolated aircraft. That class is the honest weak point of this detector.

Quantization costs **0.9 points at IoU 0.5 and 3.2 points at strict IoU**. On the synthetic corpus INT8 appeared to *beat* FP32 (0.993 vs 0.992), which was noise on a corpus too easy to separate the two; on real imagery the cost is visible and consistently in the expected direction.

```bash
python model/evaluate_detector.py --onnx model/artifacts/osp_yolov8n_int8.onnx \
    --images val/images --labels val/labels
```

### Quantization

| Metric | FP32 | INT8 | |
| :--- | ---: | ---: | :--- |
| Artifact size | 12.67 MB | 3.69 MB | 3.43x smaller |
| Latency (CPU, 640², sequential) | 94.1 ms | 51.4 ms | 1.83x faster |
| Mean relative divergence | n/a | 2.11 % | max 2.81 % |
| **mAP@0.5** (3,677 real tiles) | **0.889** | **0.880** | costs 0.9 points |
| **mAP@0.5:0.95** | **0.576** | **0.544** | costs 3.2 points |
| Bitwise determinism | n/a | PASS | identical output across runs |
| `skyroot-oam` 400 ms budget | Met | Met | 7.8x margin |

Static INT8 post-training quantization (QDQ, per-channel weights), calibrated on 32 DOTA validation tiles through the *same* preprocessing path as inference. Those 32 tiles are drawn from the same 3,677-tile split the INT8 accuracy above is scored on, so the INT8 column carries a small optimistic bias: roughly 0.9% of the scoring set was seen by the calibrator. Calibration fits activation ranges, not weights, so the effect is second-order, but it is not zero and the number is not independent. Static rather than dynamic on purpose: dynamic quantization leaves most convolutions in FP32 and makes latency data-dependent, which breaks the determinism property the assurance story rests on.

Bitwise determinism is the quiet one worth noticing: the same input produces byte-identical output across runs, which is what makes an on-orbit result reproducible on the ground.

```bash
python model/benchmark_quantization.py --platform skyroot-oam
```

### Compression

The operating-point comparison lives above, under *The compression claim is a curve, not a ratio*. What remains here is the encoding cost alone, holding content fixed.

Measured over the 20-brief DOTA corpus in `data/briefs/`:

| Comparison | Ratio | Coverage |
| :--- | ---: | :--- |
| Protobuf brief (155 B mean) vs raw tile (9.83 MB) | 63,279:1 | same area, the honest per-tile figure |
| Protobuf vs JSON | 2.66x | same content |

Protobuf over JSON buys 2.66x on identical content (413 B → 155 B mean), largely by sending a single enum byte where JSON spells out `"type": "storage-tank"` in full.

> **This ratio went up, and that is not good news.** The synthetic corpus reported 43,497:1; the DOTA corpus reports 63,279:1. Nothing about the encoding improved. These 20 tiles average 1.05 detections per brief where the synthetic scenes were denser, and an emptier brief is a smaller brief. This is the same artefact flagged under *The compression claim is a curve, not a ratio* — a headline ratio partly measures how empty the scenes happened to be, which is why the rate-distortion curve, not this number, carries the architectural argument. Reported here because it moved, not because it is evidence.

These ratios describe a single brief against a single tile. They are not a measure of how much the ground learns per byte, and quoting them as though they were is the mistake the curve exists to correct.

### Spectral bands

The stem swap replaces YOLOv8n's 3-channel first convolution with a 6-channel one (B2/B3/B4/B8/B11/B12, blue through short-wave infrared). The three visible channels inherit their pretrained weights directly; the three infrared channels are initialised to the mean of the RGB weights rather than to noise, so from the first training step the network has working edge and texture detectors on bands it has never seen. The detection head is rebuilt from 80 COCO classes to 4, carrying over the pretrained bias calibration.

The architectural motivation was that B11/B12 short-wave infrared separates man-made hull material from seawater even through light haze, where the visible bands wash out to uniform grey.

**That motivation is sound physics, and this repository cannot claim any of it.** The reason is not the corpus. It is arithmetic.

`data/synthetic_bands.py` derives every band from RGB by a fixed linear map:

```
B8  (NIR)    = 0.25·R + 0.45·G + 0.30·B
B11 (SWIR-1) = 0.80·R + 0.30·G − 0.20·B
B12 (SWIR-2) = 0.70·R + 0.20·G − 0.10·B
```

Nothing enters those lines that was not already in R, G and B. A convolution's first act is to compute weighted mixes of its input channels, so the network can form any of these for itself; being handed them pre-computed adds no information. The singular values of a derived 6-band tile show it directly — four significant components, then a cliff of roughly 60x into numerical noise:

```
1.0000  0.2341  0.1318  0.1219  0.0020  0.0006
```

Three of those components are the RGB the tile started as. The fourth is the resample applied to B11/B12 to imitate their 20 m native resolution, which is a spatial blur of a derived channel — still a function of RGB, still carrying nothing new.

On the synthetic corpus, the measurement agreed with the arithmetic: dropping B11 cost 0.001 mAP, dropping B12 cost 0.011, and dropping both cost nothing at all (0.996 to 0.996).

**On real DOTA imagery, it does not reproduce as stated.** Re-measured on the trained DOTA detector (96 held-out tiles, stride-sampled): dropping B11 alone costs 0.003 mAP (0.836 → 0.833), dropping B12 alone *improves* it by 0.007 (noise, at this sample size), and dropping **both together costs 0.045 mAP** (0.836 → 0.791) — about 5.4% relative, not zero. A control that blanks all six bands scores 0.000, so the harness is biting.

An earlier version of this file asserted that the zero-cost result "reproduces on DOTA." **That was written before any DOTA measurement existed, and it was wrong.** The arithmetic argument itself still holds — B11 and B12 are still a fixed linear map of R, G and B, so they still carry no information the visible bands did not already contain. What the real measurement shows is a *different* claim: an information-redundant channel is not the same as a channel the trained network is indifferent to losing. The stem-swap initialisation seeds the infrared channels from the mean of the RGB weights, and 32 epochs of real gradient descent evidently pulled B11 and B12 into carrying some of the load, however redundantly, rather than leaving them at zero-derivative no-ops. Losing two of six input channels at inference time is not the same operation as never having trained on them, and the two-channel joint effect is larger than either single-channel effect, which is itself a sign of a network exploiting some of the redundancy rather than none of it.

Two more numbers complicate the picture further: on real imagery, dropping **B2 (blue) costs 0.066 mAP and B4 (red) costs 0.051** — both larger than the combined SWIR cost. The bands the network is least willing to lose are two of the three plain visible ones, not the derived infrared pair the architectural motivation was about. This is a real photograph corpus behaving like one: color content carries the signal, and the theoretically-redundant channels turn out to matter less than the non-derived ones, exactly as the arithmetic predicts, just not in the specific "SWIR is free to drop" shape the earlier synthetic-only measurement suggested.

What would settle it is a sensor that measures short-wave infrared independently — and that is scarce for physical rather than editorial reasons. SWIR's wavelength is roughly three times visible light's, so the same aperture resolves roughly three times less detail, and silicon detectors cannot see SWIR at all. Sentinel-2 carries 10 m visible and 20 m SWIR; WorldView-3 resolves 31 cm panchromatic and about 3.7 m in SWIR. High-resolution short-wave infrared of ships is largely not a thing that exists to be downloaded.

So the honest position: the 6-channel stem is real engineering — channel surgery, INT8 calibration across six planes, band-dropout resilience — built correctly for a sensor this project does not have. It is not a demonstrated perception advantage, and no result here should be read as validating one.

### Latency

`51 ms per tile` is the number this file led with for a long time. It is a *mean*, it is inference only, and it was measured on a laptop with every core available. Two of those three are the wrong shape for a claim about onboard compute: a mean hides the tail a downlink budget has to absorb, and an unconstrained laptop is not a spacecraft.

Re-measured as a distribution, over 100 held-out DOTA tiles (stride-sampled, 5 warm-up runs discarded), with the whole per-tile cost split into its two parts:

| | p50 | p95 | p99 | mean |
|---|---|---|---|---|
| **Host, unconstrained** (4 cores) | | | | |
| preprocess | 56.93 | 102.48 | 226.14 | 56.88 |
| inference | 50.19 | 54.57 | 60.50 | 51.02 |
| end to end | 106.02 | 154.64 | 282.54 | 107.89 |
| **Constrained** (`--cpus 2 --memory 4g`, `OMP_NUM_THREADS=2`) | | | | |
| preprocess | 87.24 | 140.01 | 215.93 | 88.14 |
| inference | 88.07 | 113.72 | 185.76 | 82.44 |
| end to end | 168.91 | 232.71 | 307.65 | 170.58 |

All figures in milliseconds. Reproduce with `tools/latency_bench.py`; the raw output is committed under `docs/latency/`.

Three things worth reading off that table. The old 51 ms claim survives as an inference mean on an unconstrained host, which is what it always was. Halving the cores roughly doubles inference and leaves p99 end to end at **307.65 ms against the platform's 400 ms budget**, so the budget holds at the tail rather than only at the median. And preprocessing is not the free part: at two cores it costs more than inference at p50, because `rgb_to_6band()` runs two `cv2.resize` calls per tile to synthesise the SWIR bands. On a real six-band sensor that work does not exist, so the derived-band design is paying a latency cost for channels it also cannot show a perception benefit from.

This is x86 held to two cores, not ARM. It is a resource-constrained proxy and nothing more: a Pi 5 or an Orin Nano has a different instruction mix and these numbers will not extrapolate to one. That measurement still needs the hardware.

### Fault tolerance

Single-event upsets injected uniformly into the 25,026,816 bits of quantised weight memory, scored on the real DOTA detector over 96 held-out tiles (stride-sampled across the val split so all four classes are represented), 3 random draws per point.

| Weight bits flipped | Share of weight memory | mAP@0.5 | Detections emitted |
| ---: | ---: | ---: | ---: |
| 0 | 0% | 0.836 | 856 |
| 256 | 0.001% | 0.843 | 851 |
| 1,024 | 0.004% | 0.841 | 851 |
| 4,096 | 0.016% | 0.819 | 834 |
| 8,192 | 0.03% | 0.816 | 816 |
| 16,384 | 0.07% | 0.801 | 643 |
| 32,768 | 0.13% | 0.622 | 475 |
| 65,536 | 0.26% | 0.338 | 225 |
| 131,072 | 0.52% | 0.030 | 45 |
| 262,144 | 1.05% | 0.000 | 1 |
| 524,288 | 2.10% | 0.000 | 1,341 |
| 1,048,576 | 4.19% | 0.000 | 16,548 |

The detection count is the column to read, and the real detector shows the same failure shape the synthetic one did, at a different scale. Below 0.13% of weight memory flipped the model is essentially undisturbed — mAP moves inside its own run-to-run noise band, and briefly *reads higher* than baseline at 256 flips. Between 0.13% and 0.52% it collapses (0.622 to 0.030), and past 1% it goes silent-then-loud: one detection at 262,144 flips, then 1,341 at 524,288, then **16,548** at 1,048,576 — 19x the clean baseline, essentially all of it wrong, with no error raised anywhere in the stack. That collapse-then-explosion shape reproduces on the real model; only the exact flip counts at the knee moved.

This is a conditional measurement, not a radiation model. It says what survives given N flips and nothing whatsoever about how often N flips occur.

Band dropout, real DOTA detector, same 96-tile sample:

| Band dropped | mAP@0.5 | Δ from clean |
| :--- | ---: | ---: |
| none | 0.836 | — |
| B2 (blue) | 0.770 | −0.066 |
| B3 (green) | 0.847 | +0.011 |
| B4 (red) | 0.785 | −0.051 |
| B8 (NIR, derived) | 0.844 | +0.008 |
| B11 (SWIR-1, derived) | 0.833 | −0.003 |
| B12 (SWIR-2, derived) | 0.843 | +0.007 |
| B11 + B12 | 0.791 | −0.045 |
| all six *(control)* | 0.000 | −0.836 |

```bash
python resilience/degradation.py --images val/images --labels val/labels --tiles 96
```

---

## Evaluation

Four suites, because four very different things can be wrong.

92 tests locally. CI runs 84 of them and skips 8: the accuracy floor, the stem-swap check and the SEU injection tests all need a trained artifact or torch, neither of which a repository should carry. So the badge means *the deterministic layers hold*, not *the detector is accurate*. The second claim is made in Results, from a local run, and labelled as such.

`test_pipeline.py`: 16 tests over the engineering path: tensor contracts, geo-projection, protobuf round-trip, compression targets, memory budget, tile-storage equivalence, DOTA label conversion, rate-distortion accounting, and an accuracy floor on the trained detector. That last one exists because an artifact that exports cleanly, quantizes cleanly and benchmarks cleanly *while detecting nothing* passes every other test in the file.

This file is both a standalone runner and a pytest module, and for most of its life only the first half worked. Its decorator caught and reported failures rather than raising them, so under `pytest` every test in it reported PASS no matter what it asserted, including a deliberately failing probe. The outcome is now re-raised when pytest is driving. Worth stating plainly because it means the green result below is a newer claim than the tests are.

`test_resilience.py`: 33 tests over the failure behaviours. Fault injection into INT8 weights, dead spectral bands, watchdog overruns, hard model failure and corrupted briefs, plus a coverage test that fails if `AssuranceProfile` grows a field with no test that exercises it. The suite was checked by mutation, not just by running it: disabling the watchdog comparison fails two tests, and letting a degraded brief become the last-known-good fails a third.

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

Train the detector: dataset prep, stem surgery, two-phase training, ONNX export, INT8 quantization, scored on both backends.

```bash
python train.py --export          # ~100 min on CPU (26 epochs, synthetic corpus)
python train.py --quick --export  # ~3 min smoke test of every stage
```

The shipping weights are **not** from that path. They come from `tools/kaggle_train_dota.ipynb`, which prepares DOTA-v1.0 via `data/dota_prep.py` (11,046 train tiles / 3,677 val tiles at 640 px, stride 480) and runs the same two-phase schedule for 32 epochs on a Tesla P100, 3 h 11 m wall clock. Reproducing locally is the download-and-export path:

```bash
# after downloading osp_dota_artifacts.zip from the notebook's Output tab
unzip osp_dota_artifacts.zip -d .
python satellite_export.py --weights model/artifacts/osp_best.pt --calib val/images
python model/evaluate_detector.py --onnx model/artifacts/osp_yolov8n_int8.onnx \
    --images val/images --labels val/labels
```

> The notebook pins `torch==2.5.1+cu121` when it detects a compute capability below sm_70. Kaggle's preinstalled cu128 build has no kernels for the P100's sm_60, and `torch.cuda.is_available()` returns `True` anyway — so the notebook launches a real kernel to check rather than trusting that flag.

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

To check that claim rather than trust it, regenerate the corpus inside the mission container and diff it against what is committed:

```bash
python tools/verify_docker_repro.py --check-only   # prerequisites, no build
python tools/verify_docker_repro.py                # build, regenerate, diff
```

The two inputs stay out of git and out of the image: the 3.7 MB INT8 graph and the 3,677-tile held-out split are both reproducible from the notebook, so they are bind-mounted read only at run time. The claim being verified is therefore *given the artifact and the split, the container regenerates `data/briefs/`*, not *a clean clone regenerates it from nothing*. `meta.inference_ms` is excluded, being wall-clock.

The result is worth stating precisely, because it is not a clean pass. On the host, regeneration reproduces the committed corpus 20/20. In the container it reproduces 7/20 exactly; the other 13 carry the same tiles and the same detection counts, with confidences differing by at most one step on the INT8 score ladder (about 0.037). Bisecting it: the ONNX graph returns a bit-identical tensor in both environments, and so does the JPEG decode, but the derived 6-band tile does not. `rgb_to_6band()` builds the synthetic B11/B12 bands with `cv2.resize`, and `INTER_LINEAR` dispatches a different SIMD kernel against the container's CPU feature set. The disagreement is around 1e-8 relative, which against an FP32 detector would not be observable. Against an INT8 one it is: quantisation snaps that hair's width onto the next rung of a discrete score ladder, and a detection sitting on the 0.35 confidence threshold can cross it.

That is a property of quantised inference worth knowing rather than a defect in the container, so the check enforces structural agreement and reports the numeric drift with its cause. `--strict` demands bit-equality and currently fails, which is the honest state of it.

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

The same page in a container, which is what the public deployment runs:

```bash
docker build -f deploy/Dockerfile -t osp-dashboard .
docker run --rm -p 8501:8501 osp-dashboard
```

That image installs `deploy/requirements-dashboard.txt`, not the root manifest. The dashboard serves the committed corpus in `data/briefs/` rather than running the detector, so torch, ultralytics and the RAG stack are not reachable from it: dropping them takes the image from roughly 10 GB to 1.46 GB. Every import outside that slim set is wrapped in a try/except with a visible degraded state, which is what makes the trim safe rather than merely smaller. No API key is baked into the image; ORION reads one from the sidebar for the visitor's own session, so a public deployment cannot spend the owner's quota.

Verify everything:

```bash
python tests/test_pipeline.py          # 16 tests, engineering path, no API key
python -m pytest tests/test_orbital.py -v    # 43 tests, orbital mechanics + scheduler
pip install skyfield                   # needed for the independent-oracle test
```

`GEMINI_API_KEY` is needed only for the reasoning layer. Everything upstream, perception, quantization, serialization, the policy engine, runs without any key or network access.

> Training defaults to CPU deliberately. On Apple's MPS backend, the identical run that reaches `cls_loss 0.29 / mAP50 0.99` on CPU diverges to `cls_loss 6.0 / mAP50 0.02` with the same seed and the same data (observed on the synthetic corpus; the DOTA run trained on a Tesla P100 under CUDA). That's a numerical bug, not a speed trade-off.

---

Fault injection and the degradation curve:

```bash
python resilience/degradation.py          # SEU sweep + band dropout, ~4 min CPU
python -m pytest tests/test_resilience.py -v    # the failure behaviours themselves
```

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

- Not validated on satellite imagery. The detector is now trained and scored on DOTA-v1.0 aerial photography, not on drawn shapes, but aerial is not orbital: DOTA's ground sample distance is roughly 0.1-1 m against the 10 m the orbital layer models, so objects arrive one to two orders of magnitude larger than a Sentinel-2 pass would deliver them. `tools/generate_briefs.py` refuses to mint Sentinel-2 footprint fields from these tiles without an explicit `--allow-aerial-gsd`. Closing this needs real 10 m multispectral scenes with labels.
- Not a Skyroot specification. The `skyroot-oam` profile is a `DERIVED` envelope for a launch-vehicle upper-stage compute class, sized an order of magnitude below `moi-1a` so the INT8 and compression work has to genuinely matter. It is not insider knowledge of anyone's hardware and does not claim to be.
- No real-time AIS fusion, no terrestrial vessel-database integration.
- No radiation-hardening certification, no RF regulatory compliance. The SEU work in `resilience/` is a conditional measurement of what survives N bit flips, not a radiation model: it says nothing about how often N flips occur.
- No integrity check on the wire. Briefs carry no checksum, and a single flipped byte survives ingest roughly half the time as a well-formed, wrong observation. Measured, not assumed, in `test_a_single_flipped_byte_can_survive_ingest_undetected`.
- Silent model corruption is not covered by any fallback. The declared fallbacks catch failures the system can *see*. A bit-flipped model raises nothing, so nothing fires; it just returns confident nonsense, and increasingly more of it.
- The 6-band argument is unsupported, and real imagery does not rescue it, though it complicates the story. Every band is a fixed linear combination of R, G and B, so the infrared planes carry no information the visible ones did not add. On real DOTA imagery, dropping B11 and B12 together costs 0.045 mAP (not the 0.000 an earlier draft claimed before any real measurement existed) — the arithmetic redundancy argument still holds, but a trained network is not perfectly indifferent to losing channels it partially learned to lean on. Either way, the two costliest bands to drop are B2 and B4, plain visible channels, not the derived infrared pair the stem swap was motivated by. Settling the SWIR question for real needs a sensor that measures it independently. See *Spectral bands*.
- Not a link budget. Downlink capacity is rate x duration with a coarse elevation derating, not a computation from antenna gains and noise figures. See *Tradeoffs*.
- Not a licensed ground station. The Hyderabad site uses Skyroot's corporate coordinates as a planning reference, with a conservative default elevation mask.
- Pass predictions inherit TLE age. The committed snapshot is dated and graded in the UI; a stale element set gives indicative timing, not pointing-grade timing.
- No quantization-aware training: INT8 is post-training only.
- Full atmospheric correction assumed (L2A input).
