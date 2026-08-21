# Orbital Scene Preprocessor (OSP)

**On-board multispectral perception with a deterministic safety envelope around a generative reasoning layer.**

[Live command centre →](https://osp-command-centre.streamlit.app/)

![img.jpg](img.jpg)

---

## 1. The problem, stated precisely

An Earth-observation payload generates data orders of magnitude faster than it can downlink it. The usual framing is that on-board inference is a *bandwidth optimisation*. On a sufficiently constrained platform it stops being an optimisation and becomes the only thing that works at all:

| Platform profile | Contact capacity | Can it downlink one 100 MB scene? | Semantic briefs per contact |
| :--- | ---: | :--- | ---: |
| `moi-1a` (2 Mbps × 8 min) | 122.9 MB | Yes — barely, using the entire pass | 102,400 |
| `skyroot-oam` (32 kbps × 5 min) | 1.2 MB | **No.** Not one. | 1,000 |

OSP downlinks a compact structured brief instead of imagery. On the constrained profile that is the difference between one scene per *hundred* contacts and a thousand scenes per contact.

### Measured compression (`python test_pipeline.py`, T8)

| Comparison | Ratio | Coverage |
| :--- | ---: | :--- |
| Protobuf brief (226 B) vs raw tile (9.83 MB) | **43,497:1** | same area — the honest per-tile figure |
| Protobuf vs JSON | 2.4× | same content |
| Scene-level, normalised | **1,432:1** | charges the scene all 324 briefs needed to cover it |
| ~~Scene-level, unnormalised~~ | ~~463,972:1~~ | **do not quote** — one tile's brief vs a whole scene |

That last row is why this table exists. An earlier version of this README led with "85,000:1", which came from dividing a single tile's brief by a full 100 MB scene. A Sentinel-2 scene is 10980×10980 px — 324 tiles of 640×640 — so one brief covers 1/324th of the area being compared against, inflating the ratio by that factor. The normalised number is ~30× smaller and still an overwhelming argument.

> Both link budgets are `DERIVED` engineering assumptions, not operator specifications. See §6 on provenance — the repo distinguishes measured, published, and assumed numbers, and so does this README.

---

## 2. Architecture

```
 ON-BOARD (constrained)                    │  GROUND (unconstrained)
 ─────────────────────                     │  ─────────────────────
 6-band L2A tile                           │
   → stem-swapped YOLOv8n (INT8 ONNX)      │
   → per-class NMS                         │
   → geo projection                        │
   → OSP brief  (~1.2 KB JSON / protobuf) ─┼─→  RAG retrieval (FAISS)
                                           │      → episodic memory (SQLite)
                                           │      → ORION LLM reasoning
                                           │           ↓
                                           │      PolicyEngine (deterministic)
                                           │           ↓  ← holds authority
                                           │      reconciled verdict → OVV request
```

**The load-bearing design decision:** the LLM never holds authority. `agent/mission_controller.py` computes a deterministic `PolicyEngine` verdict from the detection payload alone, then reconciles the LLM's assessment against it. If the reasoning layer is unavailable, disagrees, or returns something unparseable, the policy engine's verdict stands. This is what makes a generative component admissible on a platform with an assurance requirement — and `config/platforms.py` encodes `llm_in_control_loop = False` as a profile constraint rather than a convention.

---

## 3. Perception layer

| Component | Implementation |
| :--- | :--- |
| **Spectral stem swap** | YOLOv8n's 3-channel stem replaced with 6-channel (B2/B3/B4/B8/B11/B12). RGB channels warm-start from pretrained weights; NIR/SWIR channels initialise to the RGB weight mean rather than noise, so gradient flow is healthy from epoch 0. |
| **Head re-shaping** | `Detect.cv3` classification branches rebuilt for 4 OSP classes, preserving the pretrained bias calibration on carried-over channels. |
| **Quantization** | Static INT8 PTQ (QDQ format, per-channel weights, asymmetric uint8 activations), calibrated on real 6-band tiles using the *same* preprocessing path as inference. |
| **Post-processing** | Per-class NMS via coordinate offsetting. Class-agnostic NMS deleted vessels berthed inside harbour boxes at IoU 0.73 — i.e. exactly the scenes of interest. |
| **Spectral rationale** | B11/B12 SWIR reflectance separates man-made hull material from ocean water through light haze, where visible bands wash out. |

### Reproducing the model artifacts

```bash
python train.py --export     # dataset → stem surgery → 2-phase train → FP32/INT8 export → scored
```

Runs in ~100 minutes on a CPU (26 epochs; see the MPS caveat below). `python train.py --quick` smoke-tests every stage in under 3 minutes on a tiny corpus. Individual stages remain runnable on their own — `data/synth_demo.py`, `model/train_6ch.py`, `satellite_export.py`, `model/evaluate_detector.py`.

### Measured quantization results

`python model/benchmark_quantization.py --platform skyroot-oam` regenerates every number below.

| Metric | FP32 | INT8 | |
| :--- | ---: | ---: | :--- |
| Artifact size | 12.67 MB | **3.69 MB** | 3.43× smaller |
| Latency (CPU, 640², seq.) | 110.3 ms | **62.0 ms** | 1.78× faster |
| Mean relative divergence | — | **3.52 %** | max 4.24 % |
| Bitwise determinism | — | **PASS** | identical output across runs |
| `skyroot-oam` 400 ms budget | ✗ | **✓ MET** | at 6.5× margin |

**3.69 MB does not meet the "<3 MB" target this README used to claim.** The old figure was never measured; the real number is 3.69 MB, stated as unmet rather than quietly restated.

Output shape `(1, 8, 8400)` — 4 box + 4 class channels — confirms the head is genuinely 4-class in the exported graph.

### Measured detection accuracy

Relative tensor divergence above says nothing about whether boxes survive quantization — a graph can diverge by a few percent and still lose every detection. `python model/evaluate_detector.py --onnx <model> --images osp_dataset/images/val --labels osp_dataset/labels/val` scores the deployed decision path (`inference.engine.postprocess`, same NMS, same class map) on the held-out synthetic split (80 tiles, none seen in training):

| | FP32 checkpoint | INT8 (deployed) | |
| :--- | ---: | ---: | :--- |
| mAP@0.5 | 0.992 | **0.993** | quantization cost ≈0 here |
| mAP@0.5:0.95 | 0.905 | **0.853** | quantization costs ~5 pts at strict IoU |
| ship | AP50 0.985 | AP50 0.990 | 200 instances |
| airplane | AP50 1.000 | AP50 1.000 | 48 instances |
| storage-tank | AP50 0.983 | AP50 0.982 | 116 instances |
| harbor | AP50 1.000 | AP50 1.000 | 24 instances |

Per-class numbers are reported because the composite could otherwise hide one dead class behind three strong ones — `test_pipeline.py` T13 asserts `classes_scored ≥ 3` for exactly that reason.

**This is a synthetic-tile result and should be read as one.** The corpus is procedurally generated shapes (rectangles, cruciforms, discs) on flat-colour backgrounds with a COCO-pretrained backbone — a much easier task than real Sentinel-2 imagery with genuine texture, occlusion, and sensor noise. It demonstrates the *training pipeline* produces a working detector, not real-world accuracy. See §7.

Two structural defects made this untrainable before now, independent of any tuning:

1. The synthetic corpus generated one class ("ship") while the rebuilt head emitted four. `engine.postprocess` refuses a class-count mismatch, so the two halves could never run together. `data/synth_demo.py` now generates all four classes across three composed scene archetypes (open ocean, port, airfield), including vessels berthed *inside* harbour boxes — the case per-class NMS exists for.
2. `train.py` called Ultralytics' stock trainer, whose data loader reads 8-bit RGB and cannot read 6-band float32 `.npy` tiles at all. `model/train_6ch.py` is a from-scratch two-phase loop (`v8DetectionLoss` reused, data path is OSP's own) whose preprocessing is byte-identical to `inference/engine.py:preprocess` — so what's trained is what's served.

**MPS is not used for training on this machine.** The identical run that reaches cls_loss 0.29 / mAP50 0.99 on CPU diverges to cls_loss ~6.0 / mAP50 0.02 on Apple's MPS backend with the same seed and data — a real numerical bug in this PyTorch/MPS combination, not a speed tradeoff. `model/train_6ch.py` defaults to CPU and documents the measurement; verify independently before trusting `--device mps` elsewhere.

---

## 4. Reasoning layer (ORION)

Structured generative analysis over the downlinked brief, grounded three ways:

- **RAG** — 14 curated maritime knowledge chunks (UNCLOS/EEZ, IMO AIS carriage, dark-vessel behaviour, SWIR physics, OVV trigger policy) embedded into FAISS. The index is content-fingerprinted and rebuilds automatically if the corpus drifts, because a stale vector index fails *silently and plausibly* — the worst failure mode a grounded system can have.
- **Episodic memory** — detections persist across orbital passes in SQLite, so recurring anomalies escalate rather than being re-derived from scratch each pass.
- **Constrained decoding** — the response schema is enforced by the provider's structured-output mode. Earlier versions asked the model in-prompt to avoid double quotes and then regex-scraped failures; that salvage path fabricated `conf: 0.5` values which flowed straight into the faithfulness metric.

---

## 5. Evaluation — the part that would embarrass me if it were fake

`ground/eval_suite.py` scores every brief on **six independently-failing axes**:

| Axis | Catches |
| :--- | :--- |
| `schema_validity` | structurally unusable output |
| `entity_grounding` | omissions, hallucinated detections, class substitutions (geodesic greedy matching — order-invariant) |
| `coordinate_fidelity` | positions the model generated rather than transcribed |
| `numeric_fidelity` | confidence values that were never downlinked |
| `citation_validity` | fabricated chunk IDs — **and** real IDs that were never retrieved (cited from parametric memory, which a naive checker passes) |
| `policy_consistency` | disagreement with the deterministic policy engine, penalised asymmetrically: under-escalation costs 2× over-escalation, because downgrading a real threat is the safety-critical direction |

The composite score is the **minimum** across axes, not the mean: a brief that invents coordinates is not redeemed by scoring well on schema validity.

This replaced a metric that compared `len(anomalies)` to `len(anomaly_assessments)` and returned 1.0 on equality. That metric scored a **perfect 1.0** on briefs that substituted classes, invented coordinates 400 km away, fabricated every confidence value, cited non-existent evidence, or downgraded a RED scene to GREEN. All six are now regression cases.

```bash
python -m ground.eval_suite --telemetry data/telemetry_out --live --fail-under 0.8
```

Two-tier distance gating: entity matching is generous (3 km — identify *which* detection is meant), coordinate fidelity is strict (500 m — coordinates are transcribed, not estimated).

---

## 6. Platform profiles and number provenance

`config/platforms.py` makes the deployment target data rather than scattered constants. Every field carries a provenance tag:

- `PUBLISHED` — stated publicly by the operator
- `MEASURED` — measured by us on representative hardware
- `DERIVED` — an engineering assumption, to be replaced when real specs are available

```bash
python config/platforms.py                                    # print both profiles
OSP_PLATFORM=skyroot-oam python inference/engine.py --model ... # select at runtime
```

The engine enforces the active profile: execution providers come from the profile (not "use CUDA if you can find it", which would make ground-side timings unrepresentative of flight), and briefs exceeding the link budget or tiles exceeding the latency budget are flagged.

**The `skyroot-oam` profile is a DERIVED envelope for a launch-vehicle upper-stage compute class. It is not a Skyroot specification and does not claim to be one.** It is deliberately sized an order of magnitude below `moi-1a` so that the INT8 and compression work has to actually matter.

> **Note for future me:** this repo used to be hard-wired to one operator (TakeMe2Space / MOI-1A). Rather than keeping two diverging copies of the project, that original state was frozen as branch `tm2space-original` (tag `v1.0-tm2space`) and `main` was generalised into one codebase with a swappable platform profile. `moi-1a` and `skyroot-oam` are just two entries in `PROFILES` — add a new operator by adding a new `PlatformProfile`, not by branching. Every field is tagged `PUBLISHED` (operator said so) / `MEASURED` (we tested it) / `DERIVED` (our assumption) specifically so a profile for a company we don't work for can never be mistaken for insider knowledge of their hardware.

---

## 7. Non-goals

Stated explicitly so the scope is not mistaken for a claim:

* Real-time AIS fusion or terrestrial database integration.
* Flight-hardware radiation-hardening certification.
* Full atmospheric correction (L2A input assumed).
* Quantization-aware training for on-orbit model updates.
* Encrypted RF cross-link regulatory compliance.
* **Training data.** Detection is trained on synthetic 6-band tiles and domain-adapted RGB weights, because public labelled multispectral detection datasets are scarce. Detection accuracy numbers from this repo characterise the *pipeline* — that stem surgery, training and quantization compose into something that finds real boxes — not operational performance on real Sentinel-2 imagery. The synthetic corpus is procedurally generated shapes on flat backgrounds; it is an easier task than real coastal imagery with texture, occlusion and sensor noise, and mAP on it should not be read as an estimate of real-world mAP.
* **Real-imagery validation.** The detector is trained and scores 0.99 mAP@0.5 (§3) entirely on synthetic tiles. Retraining and re-benchmarking on real Sentinel-2 scenes with real annotations is the next piece of work, not a completed one — everything downstream of detection (serialisation, policy engine, RAG, memory, eval harness) is exercised against the live detector's output as well as mock and recorded payloads, but none of it has seen a real satellite image yet.

---

## 8. Running it

```bash
conda create -n osp_dev python=3.10 -y && conda activate osp_dev
pip install -r requirements.txt

python train.py --export     # dataset → stem surgery → train → INT8 export → scored (~100 min CPU)
# or: python train.py --quick --export   # ~3 min smoke test, tiny corpus

python inference/engine.py \
  --model model/artifacts/osp_yolov8n_int8.onnx \
  --tiles osp_dataset/images/val \
  --out   data/telemetry_out \
  --platform skyroot-oam

streamlit run ground/dashboard.py
```

Requires `GEMINI_API_KEY` for the ORION reasoning layer. Everything upstream of it — perception, quantization, serialisation, the policy engine — runs without any API key.

Deployment: Dockerised Python 3.10 + ONNX Runtime, `--gpus 1 --memory 4g --cpus 2`, `/input` and `/output` mounts.
