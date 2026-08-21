# Orbital Scene Preprocessor

On-board multispectral perception with a deterministic safety envelope around a generative reasoning layer.

[Open the live command centre](https://osp-command-centre.streamlit.app/)

`0.99 mAP@0.5` · `3.69 MB INT8 model` · `62 ms per tile` · `226-byte briefs` · `43,497:1`

![OSP command centre](img.jpg)

---

## Why this exists

Start with one number.

A Sentinel-2 scene is about 100 MB. Give a spacecraft a 2 Mbps radio and an 8-minute ground pass and it can move 122.9 MB per contact, enough for exactly one scene, using the entire window.

Now shrink the radio. 32 kbps, a 5-minute pass: 1.2 MB per contact.

The scene doesn't fit. Not "fits slowly", it does not fit, and no amount of patience fixes it, because you'd need a hundred contacts to move a single image and the camera produces the next one long before that. The backlog grows faster than the link drains it, forever.

At that point, on-board inference stops being an optimisation and becomes the only thing that works at all.

| Platform profile | Contact capacity | One 100 MB scene? | Briefs per contact |
| :--- | ---: | :--- | ---: |
| `moi-1a`, 2 Mbps x 8 min | 122.9 MB | Barely, using the whole pass | 102,400 |
| `skyroot-oam`, 32 kbps x 5 min | 1.2 MB | No. Not one. | 1,000 |

OSP's answer is to never downlink the image. Detection runs on-orbit; what comes down is a structured brief: a few hundred bytes saying *what* was found, *where*, and *how confident*. The same 1.2 MB contact that could not carry a single picture carries about a thousand of these.

---

## Data flow

```
ON-BOARD (constrained)                    │  GROUND (unconstrained)
──────────────────────                    │  ─────────────────────
6-band tile  (640×640×6, 9.83 MB)         │
  → INT8 detector      62 ms              │
  → per-class NMS                         │
  → pixel → lat/lon                       │
  → protobuf brief     226 B  ────────────┼──→  RAG retrieval (FAISS)
                                          │       → episodic memory (SQLite)
                                          │       → LLM writes the analysis
                                          │              ↓
                                          │       PolicyEngine  ← holds authority
                                          │              ↓
                                          │       reconciled verdict → tasking request
```

Left of the line runs on a compute budget. Right of the line does not.

---

## Three things worth your attention

### 1. The language model is not allowed to be in charge

This is the part I'd most like a reviewer to look at.

There is something genuinely uncomfortable about putting a generative model anywhere near a spacecraft: its failure mode is *confident, fluent, plausible nonsense*. It does not crash. It does not return an error code. It returns a well-formed paragraph that happens to be wrong, and every downstream system accepts it happily.

So OSP structures around that instead of hoping it won't happen:

- `agent/mission_controller.py` computes an alert level deterministically, from the detection payload alone. No model involved. Auditable, reproducible, testable.
- The LLM produces its own independent assessment.
- `_reconcile_alerts()` compares them. When they disagree, the policy engine wins. If the LLM is unavailable, times out, or returns something unparseable, the policy verdict simply stands and the pipeline continues.
- `config/platforms.py` encodes `llm_in_control_loop = False` as a *profile constraint*, a property of the deployment target, not a convention someone can quietly refactor away.

The LLM's job is to explain and contextualise. Never to decide. That distinction is what makes a generative component admissible on a platform that has an assurance requirement at all.

### 2. Teaching an RGB network to see infrared

YOLOv8n is pretrained on ordinary colour photographs: 3 input channels, 80 output classes. Satellite multispectral data is neither.

The stem swap replaces that first convolution with a 6-channel one (B2/B3/B4/B8/B11/B12, blue through short-wave infrared). The three visible channels inherit their pretrained weights directly. The three new infrared channels are initialised to the mean of the RGB weights rather than to noise, so from the very first training step the network already has working edge and texture detectors on bands it has never seen: infrared reflectance is physically correlated with broadband visible energy, so the RGB mean is a genuinely informative prior, not a hack.

The detection head is rebuilt from 80 COCO classes down to 4 (`ship`, `airplane`, `storage-tank`, `harbor`), carrying over the pretrained bias calibration so the network doesn't start out predicting everything at 50% confidence.

Why bother with infrared at all: B11/B12 short-wave infrared separates man-made hull material from seawater even through light haze, in conditions where the visible bands wash out to uniform grey.

### 3. Every number in this file regenerates from a script

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

Two suites, because two very different things can be wrong.

`test_pipeline.py`: 13 tests over the engineering path: tensor contracts, geo-projection, protobuf round-trip, compression targets, memory budget, and an accuracy floor on the trained detector. That last one exists because an artifact that exports cleanly, quantizes cleanly and benchmarks cleanly *while detecting nothing* passes every other test in the file.

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

Launch the command centre:

```bash
streamlit run ground/dashboard.py
```

Verify everything:

```bash
python test_pipeline.py    # 13 tests, no API key required
```

`GEMINI_API_KEY` is needed only for the reasoning layer. Everything upstream, perception, quantization, serialization, the policy engine, runs without any key or network access.

> Training defaults to CPU deliberately. On Apple's MPS backend, the identical run that reaches `cls_loss 0.29 / mAP50 0.99` on CPU diverges to `cls_loss 6.0 / mAP50 0.02` with the same seed and the same data. That's a numerical bug, not a speed trade-off.

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
ground/     dashboard, 3D globe, episodic memory, LLM analyst, eval suite
```

---

## What this is not

Stated plainly so the scope isn't mistaken for a claim:

- Not validated on real imagery. Training and evaluation are entirely synthetic. Retraining on real Sentinel-2 scenes with real annotations is the next piece of work, not a finished one.
- Not a Skyroot specification. The `skyroot-oam` profile is a `DERIVED` envelope for a launch-vehicle upper-stage compute class, sized an order of magnitude below `moi-1a` so the INT8 and compression work has to genuinely matter. It is not insider knowledge of anyone's hardware and does not claim to be.
- No real-time AIS fusion, no terrestrial vessel-database integration.
- No radiation-hardening certification, no RF regulatory compliance.
- No quantization-aware training: INT8 is post-training only.
- Full atmospheric correction assumed (L2A input).
