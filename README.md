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
python data/synth_demo.py --n_train 32 --out data/input_debug   # synthetic 6-band tiles
python model/stem_swap.py                                        # 6-ch stem + 4-class head
python satellite_export.py --calib data/input_debug/images/train # FP32 export + INT8 PTQ
```

### Measured quantization results

`python model/benchmark_quantization.py --platform skyroot-oam` regenerates every number below.

| Metric | FP32 | INT8 | |
| :--- | ---: | ---: | :--- |
| Artifact size | 12.67 MB | **3.69 MB** | 3.43× smaller |
| Latency (CPU, 640², seq.) | 118.1 ms | **66.8 ms** | 1.77× faster |
| Mean relative divergence | — | **2.18 %** | max 2.49 % |
| Bitwise determinism | — | **PASS** | identical output across runs |
| `skyroot-oam` 400 ms budget | ✗ | **✓ MET** | at 6× margin |

Two honest caveats:

1. **3.69 MB does not meet the "<3 MB" target this README used to claim.** The old figure was never measured. The real number is 3.69 MB, and the target is now stated as unmet rather than quietly restated.
2. **Divergence is measured against an untrained re-headed model.** It characterises the quantization step, not detection accuracy. Accuracy retention requires a trained checkpoint — see §7.

Output shape `(1, 8, 8400)` — 4 box + 4 class channels — confirms the head is genuinely 4-class in the exported graph, which is what the previous version got wrong.

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
* **Training data.** Detection is trained on synthetic 6-band tiles and domain-adapted RGB weights, because public labelled multispectral detection datasets are scarce. Detection accuracy numbers from this repo characterise the *pipeline*, not operational performance on real Sentinel-2 imagery.
* **A trained detector.** The stem swap and 4-class head rebuild are verified in the exported graph, and the quantized model runs end-to-end at 66.8 ms — but the re-headed classification branches have not been trained, so `engine.py` currently emits **zero detections** on synthetic tiles. Everything downstream (serialisation, policy engine, RAG, memory, eval harness) is exercised against mock and recorded payloads. Training is the next piece of work, not a completed one.

---

## 8. Running it

```bash
conda create -n osp_dev python=3.10 -y && conda activate osp_dev
pip install -r requirements.txt

python data/synth_demo.py --n_train 20 --out data/input_debug
python model/stem_swap.py
python satellite_export.py --calib data/input_debug/images/train

python inference/engine.py \
  --model model/artifacts/osp_yolov8n_int8.onnx \
  --tiles data/input_debug/images/train \
  --out   data/telemetry_out \
  --platform skyroot-oam

streamlit run ground/dashboard.py
```

Requires `GEMINI_API_KEY` for the ORION reasoning layer. Everything upstream of it — perception, quantization, serialisation, the policy engine — runs without any API key.

Deployment: Dockerised Python 3.10 + ONNX Runtime, `--gpus 1 --memory 4g --cpus 2`, `/input` and `/output` mounts.
