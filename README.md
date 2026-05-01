# 🛰️ Orbital Scene Preprocessor

> Target Platform: MOI-1A (100TOPS GPU / 4GB VRAM / OrbitLab Environment)

---

## 1. Executive Summary: "The Semantic Gateway"
OSP transforms MOI-1A from a passive sensor into an **Active Analyst**. By shifting compute-heavy perception to the edge, OSP delivers a compression ratio of ~85,000:1 (downlinking a 1.2KB JSON brief instead of a 100MB+ raw file). This provides >99.99% bandwidth reduction per scene at near-zero RF cost.

---

## 2. Product Vision
Turn OrbitLab into a Semantic API for Earth Intelligence.

* On-Board: 6-band quantized feature extraction.
* Downlink: Minimalist metadata "tokens."
* Ground-Side: 2D Situational Awareness map + LLM reasoning.
* Spectral Logic: Exploits Short-Wave Infrared (SWIR) reflectance contrast (B11/B12) to identify man-made vessel materials that exhibit distinct spectral fingerprints compared to ocean water, even in low-visibility conditions.

---

## 3. Functional Requirements

| Feature | Requirement | Implementation Detail |
| :--- | :--- | :--- |
| **Spectral Detector** | 6-band object detection (B2, B3, B4, B8, B11, B12). | YOLOv8n with 6-channel stem-swap; INT8 quantized. Post-processing (NMS) runs on-board for clean JSON output. |
| **Spectral Handling** | Utilize NIR (B8) and SWIR (B11/12). | B11/B12 resampled to 10m via bilinear interpolation for spatial alignment with RGB/NIR grid. Exploits SWIR reflectance contrast for dark-ship detection through light atmospheric haze |
| **JSON Schema** | LLM-ready <2KB payload. | Includes: `scene_id`, `cloud_cover`, `anomalies: [{type, lat_lon, conf}]`. |
| **Command Center** | 2D Interactive Map Dashboard. | **Streamlit + Leaflet/Folium**: Visualizes tile footprints and anomaly pins. |
| **LLM Reasoning** | Ground-side "Analyst" wrapper. | **Gemini 1.5 Pro**: Parses JSON to generate risk-weighted alerts. |

---

## 4. OVV API Contract & Deployment Spec

### OVV API Contract (Closed-Loop Verification)
* Request (Ground → Sat): `{"request_id": "REQ-001", "target_coords": [lat, lon], "reason": "high_uncertainty", "priority": 1}`
* Response (Sat → Ground): `{"status": "scheduled", "eta_minutes": 92, "payload_format": "256x256_crop_base64"}`

### Deployment Specification (OrbitLab Container)
* Image: Dockerized Python 3.10 / ONNX Runtime-GPU.
* Resource Caps: `--gpus 1 --memory 4g --cpus 2`.
* Mount Points: `/input` (Source L2A tiles), `/output` (JSON telemetry).
* Throughput: Batch mode processing; target latency **<800ms** per 1km² tile at INT8.

---

## 5. Non-Goals
* Real-time AIS fusion or terrestrial database integration.
* Flight-hardware radiation hardening certification.
* Full-scale atmospheric correction engine (L2A assumed).
* Dynamic quantization-aware training (QAT) for on-orbit updates.
* Regulatory compliance for encrypted RF cross-links.
* Training Data Scope: Simulates multispectral inference using domain-adapted RGB weights (channel mapping) due to current scarcity of public labeled multi-spectral detection datasets.

---

## 6. Success Metrics

| Metric | Target | Why TM2S Cares |
| :--- | :--- | :--- |
| Compression Ratio | >99.99% (85,000:1) | Reduces RF downlink load, maximizing the value of $2/min OrbitLab compute. |
| Model Size | <3MB (INT8) | Enables concurrent co-location with other OrbitLab user apps. |
| Inference Latency | <800 ms (INT8) | Ensures real-time anomaly flagging during orbital pass. |
| Reproducibility   | Deterministic Execution | Critical for mission-assurance and ground-side debugging. |

---

> understand the full stack:
> * Satellite constraints (VRAM, latency, power)
> * Spectral science (why 6 bands matter, not just 3)
> * Data economics (cost per inference, bandwidth saved)
> * LLM integration (structured reasoning, not just "summary")
> * System design (OVV protocol, closed-loop commanding)
