# Architecture

The README is the argument: here is what OSP does and here are the numbers. This document is the derivation: here is why the system has this shape, rebuilt from the constraint outwards, with the forks in the road left visible.

Nothing here is a summary of the README. Where the two overlap, this file goes one level lower: to the byte layout, to the line of code, to the measurement that decided it. Where the repository has retracted a claim, the retraction is kept and the mechanism of the original error is explained, because those are the most instructive parts of the project.

## How to read the numbers

Every figure in this document carries two things: the command that regenerates it, and a provenance tag borrowed from `config/platforms.py:39`.

| Tag | Means |
| :--- | :--- |
| `PUBLISHED` | An operator stated it in public material |
| `MEASURED` | Measured on this repository, by a script named alongside it |
| `DERIVED` | An engineering assumption or arithmetic from other numbers, to be replaced when real specs exist |

Two states of the repository matter when checking anything below.

`model/artifacts/` and `val/` are gitignored (`.gitignore:73`, `.gitignore:85`). The INT8 graph is 3.7 MB and the held-out split is 3,677 tiles at 396 MB, both reproducible from `tools/kaggle_train_dota.ipynb`. So a clean clone can run the deterministic layers and cannot run the detector. That distinction is load bearing and it recurs throughout.

The full suite is green today: `92 passed in 33.44s` (`MEASURED`, `.venv/bin/python -m pytest tests/ -q`, 2026-08-24, on a machine that has the artifact, the split, and `skyfield` installed). CI runs the same 92 and skips the 8 that need a trained artifact or torch, so the badge means the deterministic layers hold, not that the detector is accurate.

---

## 1. The forcing constraint

Take Sentinel-2C's actual element set from CelesTrak, propagate it with SGP4, and ask when it clears a 10 degree elevation mask over Hyderabad. You get an answer, and the answer is not a round number:

```
AOS 2026-08-21 16:43:43.946348+00:00   duration 10.155 min   peak 57.74 deg (good)
theoretical 2,437,197 B   efficiency 0.80   usable 1,949,757 B
usable passes in the following 24 h: 2
```

`MEASURED`, from `orbital/passes.py:find_passes` on the committed snapshot `data/tle/celestrak_resource_2026-08-21.tle`, planned from the last capture time in `data/briefs/manifest.json`. The dashboard computes exactly this in `ground/dashboard.py:557`.

1.95 MB. That is the whole budget. Now price what it would have to carry.

One tile of a scene, held in memory as the pipeline holds it, is:

```
640 px x 640 px x 6 bands x 4 bytes (float32) = 9,830,400 B = 9.83 MB
```

**That figure is not a downlink cost, and for most of this project's life it was used as one.** It is a working-set size. Nothing transmits an uncompressed float32 buffer; a spacecraft compresses first, and the standard it compresses with is CCSDS 123.0-B-1. Priced that way, over the committed corpus (`MEASURED`, `data/briefs/manifest.json` `raw_ccsds_bytes`, encoder at `ground/ccsds123.py`):

```
mean over 20 held-out tiles = 12,234,137 B / 20 = 611,707 B = 0.61 MB per tile
                                                = 1.99 bits per sample
```

A factor of 16 sits between those two numbers, and every compression claim this project made used the larger one. Section 9 records what that cost. `orbital/downlink.py` now defaults to the wire price, `RAW_TILE_BYTES_CCSDS`, and keeps `RAW_TILE_BYTES_FLOAT32` named separately so the distinction stays visible instead of implied.

A scene covers 110 km on a side and a tile 6.4 km, so a scene is `(110 / 6.4)^2` = **295.4 tiles**, or **180.7 MB** sent losslessly. (The ~100 MB usually quoted for a Sentinel-2 product, `DERIVED`, hardcoded at `inference/serialization_utils.py:344`, is a *delivered* product: already lossily compressed, not every band at full rate. It is not the like-for-like number and is no longer on the critical path of this argument.)

So the arithmetic that shapes everything downstream is a single division:

```
                 one contact:  1,949,757 B
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
  one 180.7 MB scene          one 0.61 MB tile          one 155 B brief
   0.0108 of it fits          3.19 of them fit          12,579 of them fit
        │                           │                           │
   "come back in                "three tiles              "and 99.6% of the
    93 contacts"                 out of 295"              window is still free"
```

The left column is the one that kills the naive design. Three tiles a contact, six a day, against a scene that is 295: a spacecraft imaging anything at all produces backlog faster than the link drains it, forever. There is no patience strategy, and there is no better compression of pixels that closes a factor of 93 on a scene while the sensor keeps running. CCSDS 123 *is* the better compression of pixels, and it is already in that middle column.

That is why on-board inference is not an optimisation here. It is the only operating point that exists.

Run the real corpus through that real window and the comparison is not close:

> **20 held-out tiles as raw imagery.** 12,234,137 B under CCSDS 123. Against this window's 1.95 MB, that is **6.28 contacts**, about three days at 2 usable passes per day. (`MEASURED`, `DownlinkPlan.raw_downlink_passes`, `orbital/downlink.py`, against `RAW_TILE_BYTES_CCSDS`.)
>
> **The same 20 tiles as briefs.** 8,266 B of JSON, 21 detections. **One** pass, using 0.42% of it, with room for 4,700 more briefs. (`MEASURED`, `data/briefs/manifest.json` totals, plus `DownlinkPlan.capacity_in_briefs`.)

### The number that moved, and was not updated everywhere

The README's *Why this exists* section states this comparison as "15.8 KB, same 104 detections, 19,863x fewer bytes", and its section 2 notes that "all five deferrals are `oversize-brief`". Neither reproduces against the corpus currently committed.

Both are true of the *previous* corpus. `git show 55dd973:data/briefs/manifest.json` reports 104 anomalies and 15,840 wire bytes, with 5 of its 20 briefs over the 1024 B per-payload cap and 9,898 B actually scheduled, which is exactly 196.6 MB / 9,898 B = 19,863. That corpus was drawn from synthetic drawn-shape tiles. The DOTA regeneration at `3f93609` replaced it with 21 anomalies and 8,266 B, and against the current corpus the scheduler defers nothing at all: **20 of 20 briefs fit, 0 deferrals.**

So the mechanism behind the stale figure is not sloppiness about arithmetic. It is that a headline was derived from a regenerable artifact, the artifact was regenerated, and the derived text was not re-derived with it. The lesson generalises past this repository: a number that is computed from committed data should either be generated into the document or asserted by a test, because prose does not have a dependency graph.

Both claims regenerate honestly. Both differ from what the README says. Stated here rather than quietly corrected, because the correction is the interesting part.

### Where the budget comes from

```python
# config/platforms.py:70
@property
def bytes_per_contact(self) -> float:
    return self.downlink_kbps * 1000 / 8 * self.contact_minutes_per_orbit * 60
```

Three lines, and every constant in them is tagged. `SKYROOT_OAM` (`config/platforms.py:161`) declares 32 kbps, 5.0 contact minutes per orbit, 1024 max payload bytes, all `DERIVED`. `MOI_1A` declares 2048 kbps and 8 minutes, also `DERIVED`, against a `PUBLISHED` compute budget.

The `contact_minutes_per_orbit = 5.0` constant is the interesting one, because the orbital layer now contradicts it. The real geometry says a good Hyderabad pass runs 10.155 minutes, so the guess was pessimistic by roughly 2x. Pessimistic is the useful direction to be wrong in, but it was still a guess, and it is now a computation that sits beside the guess rather than replacing it: `ground/rate_distortion.py:337` still prices its contact from the profile constant (1,200,000 B) while `ground/dashboard.py:557` prices its plan from the propagated window (1,949,757 B). Two numbers, two provenances, both labelled. If you see 1.2 MB in one place and 1.95 MB in another, that is why.

> ### The interview answer
>
> "Everything in this project falls out of one division. A Sentinel-2 scene is a hundred megabytes. I took Sentinel-2C's real element set, propagated it with SGP4, and computed when it actually clears a ten degree mask over Hyderabad: two usable passes a day, the next one ten minutes long. At thirty-two kilobits, derated for the pass geometry, that is 1.95 megabytes. So one contact buys you two percent of a scene, or twenty percent of a single six-band tile. You cannot move one tile. And the camera keeps producing them, so the backlog grows faster than the link drains it, permanently. At that point on-board inference stops being an optimisation and becomes the only thing that works. Every decision downstream, the wire format, the quantization, the scheduler, is a consequence of that one number."

**Concepts you now own.** *Workload characterisation before architecture.* You derive the system's shape from the binding resource constraint rather than from the technology you wanted to use. This is the framing move at the front of *Designing Machine Learning Systems* chapter 2: translate the business or mission objective into an ML objective and its constraints first, because the constraint determines whether an ML system is even the right shape. Outside orbital work it shows up identically in on-device inference, in edge video analytics, and anywhere egress cost rather than compute is the limit.

---

## 2. The wire format, derived

Here is a real brief off the wire, `data/briefs/P0019_00000_01920.json`, encoded through `inference/serialization_utils.py:serialize_to_binary`:

```
0a1150303031395f30303030305f30313932301218323032362d30382d32315430353a
34363a30332e3739325a1a2409d5cbef349921254011622cd32f113f2540191ea7e848
2e25524021516a2fa2ed285240256f12833a2a23080411cfda6d179a33254019d39ffd
4811285240255305133f2a08c703c1019304b1023500004c423a136f73702d796f6c6f
76386e2d696e74382d763140a8bb01
```

155 bytes. That is one imaged tile, one detected object, its class, its position on the Earth, its confidence, its pixel box, the model that produced it, and how long it took. The whole observation.

Now derive why it looks like that.

### What must a spacecraft say?

Start from the operator, not from the schema. The ground needs to answer: *where did you look, what did you see, how sure are you, and can I trust this run?* Four questions, and each maps to a field group.

```
scene_id + timestamp_utc + tile_footprint   →  where and when did you look
anomalies[]                                 →  what did you see
confidence per anomaly                      →  how sure are you
model_version + inference_ms + cloud_cover  →  can I trust this run
```

Everything else is not the spacecraft's job. Risk levels, alert colours, tasking recommendations, prose: all of that is ground-side reasoning over these fields, and none of it survives contact with a byte budget. The brief is deliberately the smallest thing the ground cannot reconstruct on its own.

### The wrong version first

The obvious encoding is the one the dashboard already reads, JSON:

```json
{"scene_id":"P0019_00000_01920","timestamp_utc":"2026-08-21T05:46:03.792Z",
 "tile_footprint":{"lat_min":10.565622,...},"cloud_cover":0.001,"anomaly_count":1,
 "anomalies":[{"type":"harbor","lat_lon":[10.600785,72.626055],"conf":0.5743,
 "bbox_px":[455,193,531,305]}],"meta":{...}}
```

410 bytes for the same content. Where does the extra 255 go? Almost entirely into three habits JSON cannot break: it spells every field name out on every message, it writes numbers as decimal text, and it writes class names as words. `"type":"harbor"` is 15 characters. The same fact in protobuf is `0804`: two bytes, one for the tag and one for the enum value.

Measured across the whole committed corpus (`MEASURED`, 20 briefs, script in the repro block below): protobuf mean **155.35 B**, JSON mean **413.30 B**, ratio **2.66x**, and against the 0.61 MB CCSDS-compressed tile, **3,938:1**.

### Field by field

```protobuf
// osp.proto:24
option optimize_for = LITE_RUNTIME;
```

`LITE_RUNTIME` drops the descriptor pool and the reflection machinery from generated code. You lose `text_format`, `Any`, and anything that introspects a message at runtime. You gain a much smaller generated module and no dynamic type registry on a rad-tolerant SoC with 1 GB of memory. The trade is honest for a payload that only ever serialises and parses one message type it was compiled against.

```protobuf
// osp.proto:28
enum AnomalyType { UNKNOWN = 0; SHIP = 1; AIRPLANE = 2; STORAGE_TANK = 3; HARBOR = 4; }
```

An enum rather than a string, for two reasons that are not the same. The byte saving is the obvious one. The stronger one is that an enum makes an unrepresentable class unrepresentable: a corrupted string field yields a plausible unknown class name, while a corrupted enum yields an integer the ground maps to `unknown` at `inference/serialization_utils.py:86`. `UNKNOWN = 0` exists because proto3 requires the zero value, and it doubles as the sentinel for exactly that case.

```protobuf
// osp.proto:41
double lat_min = 1;
// osp.proto:52
float confidence = 4;
```

Two different float widths in one message, and the file states its reason at `osp.proto:11`: "float32 has only ~7 decimal digits of precision, giving ~11m error at the equator."

That reason does not survive checking. Round-tripping every coordinate in the committed corpus through float32 gives a worst-case error of **0.126 m** (`MEASURED`). The worst case anywhere on Earth is at longitude 180, where float32's ulp is 1.526e-5 degrees, about **1.7 m**. The 11 m figure comes from reading "7 decimal digits" as 7 significant digits of a three-digit number, which is roughly an order of magnitude pessimistic against the actual binary representation.

So the *stated* justification is wrong and the *decision* is still defensible, for a different reason: at 10 m ground sample distance a 1.7 m worst case is 17% of a pixel, and paying 8 bytes per anomaly plus 16 bytes per footprint to never think about coordinate precision again is cheap insurance on a 155 byte message. Worth knowing what it costs, though: on a one-anomaly brief that is 24 of 155 bytes, 15% of the payload, spent on precision the sensor cannot deliver.

`confidence` as float32 is the mirror image and is unambiguously right. A detector score is a noisy quantity with about two meaningful digits. Its float32 round trip on this corpus is exact to 1e-7, which is four orders of magnitude finer than the INT8 score ladder the number came off.

```protobuf
// osp.proto:56
repeated int32 bbox_px = 5;
```

Proto3 packs repeated scalars, so `[455, 193, 531, 305]` encodes as `2a08` followed by four varints of two bytes each: 10 bytes total. The file notes at `osp.proto:9` that a fixed sub-message would save two bytes of tag overhead and that readability won.

There is a sharper edge here that the comment does not mention. `int32` varints sign-extend to 64 bits, so a negative value costs **10 bytes instead of 2**:

```
varint bytes for  455 : 2
varint bytes for   -3 : 10
```

`MEASURED`. No coordinate in the committed corpus is negative (84 bbox values, max 640, min 0), but nothing prevents one: `postprocess` converts centre-width boxes to corners at `inference/engine.py:213` and writes `int(b[0])` straight out at `inference/engine.py:340` with no clamp to the tile. A YOLO box whose left edge decodes past the image border produces exactly that. One such coordinate adds 8 bytes to a 155 byte message, and four of them add 32. The fix is one line, `sint32` with zigzag encoding, or a clamp in `postprocess`. Neither is done today.

### The actual byte layout

```
off   bytes     field                                          size
────────────────────────────────────────────────────────────────────
  0   0a 11     f1  LEN=17   scene_id        "P0019_00000_01920"   19
 19   12 18     f2  LEN=24   timestamp_utc   "2026-08-21T05:46:03.792Z"
                                                                   26
 45   1a 24     f3  LEN=36   tile_footprint                        38
        47   09    f1 F64    lat_min  10.565622                     9
        56   11    f2 F64    lat_max  10.623178                     9
        65   19    f3 F64    lon_min  72.580950                     9
        74   21    f4 F64    lon_max  72.639504                     9
 83   25        f4  F32      cloud_cover     0.001                   5
 88   2a 23     f5  LEN=35   anomalies[0]                          37
        90   08 04 f1 VARINT type = 4 (HARBOR)                      2
        92   11    f2 F64    lat  10.600785                         9
       101   19    f3 F64    lon  72.626055                         9
       110   25    f4 F32    confidence 0.5743                      5
       115   2a 08 f5 LEN=8  bbox_px [455, 193, 531, 305]          10
125   35        f6  F32      inference_ms    51.0                    5
130   3a 13     f7  LEN=19   model_version   "osp-yolov8n-int8-v1"  21
151   40 a8bb01 f8  VARINT   compression_ratio 23976                 5
────────────────────────────────────────────────────────────────────
                                                       total       155
```

Read the totals column and the architecture of the format falls out:

**The envelope is about 115 bytes and one detection costs about 36.** Check it against the whole corpus, grouped by detection count (`MEASURED`): empty briefs measure 113 to 118 B, one-anomaly briefs 148 to 155 B, two-anomaly briefs 189 to 192 B. The few bytes of spread are varint width, on `compression_ratio` and on the bbox coordinates. So this format is dominated by fixed overhead until roughly four detections per tile, and the marginal cost of finding something is small. That is the right shape for the mission, where most tiles are empty ocean and the interesting ones are not much bigger. It is the wrong shape for a dense-scene sensor, where you would move the footprint and model version into a per-pass header and send anomalies as a stream.

**60 of 155 bytes are strings.** `scene_id`, `timestamp_utc` and `model_version` are 66 bytes with their tags. A scene id derived from the tile grid position, a timestamp as a varint offset from a campaign epoch, and a model version as a two-byte registry id would take that to under 12 bytes and cut the envelope by a third. It has not been done because none of the three is on the critical path of the argument, and readability of a committed corpus is worth real bytes when the corpus is a review artifact.

**`anomaly_count` does not exist on the wire.** The JSON carries it (`inference/engine.py:112`), protobuf does not, and `_proto_to_json_str` reconstructs it from `len(brief.anomalies)` at `inference/serialization_utils.py:242`. This matters more than it looks: `BriefCandidate.from_payload` rejects a brief whose count and list disagree (`orbital/downlink.py:200`), a corruption that is simply **not representable** in the protobuf encoding. The redundant field is a JSON-only failure mode, and removing redundancy removed a class of corruption with it.

### Two things this format cannot do

First, and this is the sharpest gap in the project: **`degraded` is not on the wire.**

`inference/engine.py:102` adds `degraded`, `fallback_action` and `fault` to a payload when the perception path fails, and `OSPPayload.to_json` emits them at `inference/engine.py:118`. `osp.proto` has no such fields, and `payload_to_proto` does not map them. Demonstrated:

```
engine JSON : {..."degraded":true,"fallback":{"action":"hold_last_known_good_and_flag_ground",
               "fault":"RuntimeError: simulated"},"meta":{...}}
after proto : degraded=False  fallback_action=None  fault=None
```

`MEASURED`. A degraded brief round-tripped through the declared wire format arrives on the ground indistinguishable from a fresh observation. The entire fallback story in section 7 rests on that flag being unmistakable, and it is unmistakable on the JSON path, which is what `run_batch` writes, what `tools/generate_briefs.py` commits, and what the scheduler ingests. It is silently dropped by the format the 155 byte figure is measured in.

Second, and consequently: **protobuf is not on any runtime path today.** Grep for `serialize_to_binary` outside its own module and every hit is in `tests/test_pipeline.py`. The engine writes JSON, the dashboard reads JSON, the scheduler prices JSON (`orbital/downlink.py:214`), and `ground/rate_distortion.py:146` explicitly prices briefs as minified JSON and calls the protobuf figure the conservative one. So `osp.proto` is a designed, tested, measured format that describes what would ship, and the operational corpus is 2.66x larger than the headline. The rate-distortion result is charged the larger number, so the architectural argument is not affected. The 155 byte claim is a claim about the format, not about the file on disk.

One more piece of drift worth naming: `osp.proto:64` still documents a "compression ratio ~250,000-500,000:1", and `inference/engine.py:27` still says "~85,000:1, the headline PRD figure". Both are the retracted coverage-mismatched figure from section 9. The class that computes it now labels it `proto_vs_raw_scene_unnormalised` and documents the 324x error in its own docstring (`inference/serialization_utils.py:264`), which is exactly right. The header comments were not updated with it.

Reproduce every number in this section:

```bash
python - <<'PY'
import json, glob
from inference.engine import OSPPayload, Anomaly
from inference.serialization_utils import serialize_to_binary
def load(b):
    return OSPPayload(scene_id=b["scene_id"], timestamp_utc=b["timestamp_utc"],
        tile_footprint=b["tile_footprint"], cloud_cover=b["cloud_cover"],
        anomalies=[Anomaly(a["type"], *a["lat_lon"], a["conf"], a["bbox_px"])
                   for a in b["anomalies"]],
        inference_ms=b["meta"]["inference_ms"],
        model_version=b["meta"]["model_version"],
        compression_ratio=b["meta"]["compression_ratio"])
p = [len(serialize_to_binary(load(json.load(open(f)))))
     for f in sorted(glob.glob("data/briefs/P*.json"))]
print("proto mean", sum(p)/len(p), "raw tile ratio", 611707/(sum(p)/len(p)))
PY
```

> ### The interview answer
>
> "The brief is 155 bytes of protobuf and I can walk you through every one of them. Sixty of them are strings: scene id, timestamp, model version. Thirty-eight are the tile footprint as four float64s. Thirty-seven are the one detection: class as a two-byte enum instead of the fifteen characters JSON spends on the word `harbor`, lat and lon as float64, confidence as float32 because a detector score has two meaningful digits, and the pixel box as four packed varints. The shape that falls out is a 118-byte envelope plus 37 bytes per detection, which is the right shape when most tiles are empty. Two honest gaps: the schema's stated reason for float64 coordinates overstates float32's error by about ten times, and the degraded flag that the whole fault-tolerance story depends on exists in the JSON encoding and not in the protobuf one, so a degraded brief round-tripped through the declared wire format comes back looking healthy."

**Concepts you now own.** *Wire-format design under a byte budget*, and specifically the discipline of pricing each field against what it buys. The generalisable moves: enums instead of strings at trust boundaries, matching numeric width to the precision the sensor can actually deliver, understanding your encoder's variable-length integers well enough to know that a sign can cost you 8 bytes, and removing redundant fields because redundancy is a corruption surface. This is the concrete face of *DMLS* chapter 3 on data formats, text against binary, and it applies unchanged to any high-volume telemetry, event log, or feature-store schema where per-record bytes multiply by billions.

---

## 3. Perception under a budget

Here is the number that decides the whole storage design. The DOTA training corpus is roughly 20,000 tiles. Materialise each as the `(640, 640, 6)` float32 array the model consumes and you get:

```
20,000 x 9.83 MB = ~200 GB
```

Store the same tiles as the RGB JPEGs they came from and derive the six bands on read, and you get about **2 GB**. A hundred to one, for a derivation that is a fixed linear map plus a resample and costs microseconds.

That trade is the entire reason `data/tiles.py` exists, and its docstring says so at `data/tiles.py:26`. But look at what it caused, because that is the more interesting half.

### The bug that a storage decision creates

Before `data/tiles.py`, six different consumers opened tiles themselves with `sorted(dir.glob("*.npy"))` and `np.load`: the training loader, the evaluator, the INT8 calibrator, the quantization benchmark, the inference engine and the brief generator. That was fine while `.npy` was the only form. The moment JPEG tiles appeared, five of the six either raised "no tiles found" or, worse, scored an empty directory as a detector that found nothing.

The second failure is the dangerous one, and it recurs in this project like a refrain: *a pipeline that silently produces an empty result looks exactly like a pipeline that works on an easy input.* The fix is one module that owns the question, plus a test that pins it (`tests/test_pipeline.py:975`, `test_tile_format_equivalence`), which writes the same source pixels in both forms and asserts the arrays agree exactly. PNG rather than JPEG in that test, because lossless is the only way "exactly" is a meaningful word.

`read_tile` refuses to return a zero array on a decode failure (`data/tiles.py:72`), for the same reason: a blank tile scores as "found nothing", which is indistinguishable from a genuinely empty scene and would corrupt an accuracy number invisibly.

### The six bands, and the arithmetic that undermines them

```python
# data/synthetic_bands.py:105
b8  = np.clip(0.25 * r + 0.45 * g + 0.30 * b, 0.0, 1.0)
raw_b11 = np.clip(0.80 * r + 0.30 * g - 0.20 * b, 0.0, 1.0).astype(np.float32)
raw_b12 = np.clip(0.70 * r + 0.20 * g - 0.10 * b, 0.0, 1.0).astype(np.float32)
```

Then B11 and B12 are downsampled by two with `INTER_AREA` and bilinearly upsampled back (`data/synthetic_bands.py:114`), to imitate Sentinel-2's 20 m native SWIR grid on the 10 m visible grid.

Nothing enters those three lines that was not already in R, G and B. A convolution's first act is to compute weighted mixes of its input channels, so the network can form any of these itself. Handing them over pre-computed adds no information, and you can see it directly in the singular values of a derived tile:

```
P0007_00000_01497   1.0000  0.0976  0.0550  0.0461  0.0003  0.0000   cliff 147x
P0961_01920_00471   1.0000  0.1728  0.0837  0.0543  0.0006  0.0000   cliff  93x
P1854_01440_01920   1.0000  0.1318  0.0549  0.0202  0.0003  0.0000   cliff  61x
P2794_00000_03360   1.0000  0.1764  0.0538  0.0297  0.0003  0.0000   cliff  88x
```

`MEASURED` on real DOTA tiles, mean-centred, normalised to the leading value. Four significant components then a cliff of 60x to 200x into numerical noise. Three of the four are the RGB the tile started as. The fourth is the resample applied to B11 and B12, which is a spatial blur of a derived channel: still a function of RGB, still carrying nothing new.

**And yet the trained network is not indifferent to losing them.** Re-measured on real DOTA imagery over 96 stride-sampled held-out tiles, dropping B11 alone costs 0.003 mAP, dropping B12 alone *improves* it by 0.007 (noise at this sample size), and dropping **both together costs 0.046 mAP**, 0.836 to 0.791, about 5.4% relative.

Those two facts are compatible, and the reconciliation is the point. *Information-redundant is not the same as a trained network being indifferent to losing the channel.* The stem swap seeds the infrared channels from the mean of the RGB weights, which is a live starting point rather than zero, and 32 epochs of real gradient descent evidently pulled B11 and B12 into carrying part of the load, however redundantly. At inference time, zeroing a channel the network learned to lean on is a distribution shift, not an information loss. Losing two of six input channels is not the same operation as never having trained on them.

The two costliest bands to drop are **B2 (blue), 0.066 mAP, and B4 (red), 0.051 mAP**: two of the three plain visible channels, not the derived infrared pair the architecture was motivated by. A control that blanks all six scores 0.000, which is how you know the harness bites.

An earlier version of the README asserted the zero-cost result "reproduces on DOTA". That was written before any DOTA measurement existed, and it was wrong. What settles the SWIR question is a sensor that measures short-wave infrared independently, and that is scarce for physical reasons: SWIR's wavelength is roughly three times visible light's, so the same aperture resolves roughly three times less detail, and silicon detectors cannot see SWIR at all. Sentinel-2 carries 10 m visible and 20 m SWIR. WorldView-3 resolves 31 cm panchromatic and about 3.7 m in SWIR. High-resolution short-wave infrared of ships is largely not a thing that exists to be downloaded.

The honest position: the 6-channel stem is real engineering, built correctly, for a sensor this project does not have.

There is a second cost, and it is not accuracy. Deriving bands is two `cv2.resize` calls per tile, and at two cores that preprocessing costs **more than inference at p50** (87.24 ms against 88.07 ms, and 88.14 ms mean against 82.44 ms). On a real six-band sensor that work does not exist. The derived-band design is paying a latency cost for channels it also cannot show a perception benefit from.

### Stem surgery

YOLOv8n's stem is `nn.Conv2d(3, 32, k=3, s=2, p=1)`. You need `nn.Conv2d(6, 32, ...)`. The naive version initialises the three new input channels with Xavier, which is what the module docstring at `model/stem_swap.py:13` still says it does. **The code does something else:**

```python
# model/stem_swap.py:136
new_conv.weight[:, :3, :, :] = old_weight
...
# model/stem_swap.py:150
rgb_mean = old_weight.mean(dim=1, keepdim=True)  # [32, 1, 3, 3]
new_conv.weight[:, 3, :, :] = rgb_mean.squeeze(1)   # B8  NIR
new_conv.weight[:, 4, :, :] = rgb_mean.squeeze(1)   # B11 SWIR-1
new_conv.weight[:, 5, :, :] = rgb_mean.squeeze(1)   # B12 SWIR-2
```

The inline comment at `model/stem_swap.py:143` gets it right and the header docstring is stale. Flagged because a reader who trusts the docstring will believe a different thing about the model than what is on disk.

The reason RGB-mean beats Xavier is worth stating precisely, because "warm start" is too vague. A pretrained stem's 32 filters are edge, corner and texture detectors, and each is a specific spatial pattern applied across three input channels. Averaging over the channel axis keeps the *spatial* structure of every filter and discards only the colour selectivity. So at step zero the network already has working edge detectors on bands it has never seen, and the activation magnitudes on the new channels sit in the same range as the old ones, which keeps the first BatchNorm's statistics sane. Xavier gives you noise with the right variance and no structure at all, so the first few hundred steps are spent relearning edges the model already knew.

The test that pins it (`tests/test_pipeline.py:262`) asserts the RGB channels are preserved exactly, the three new channels equal the RGB mean, and that the new channels' standard deviation does not exceed the old channels': the mean of three noisy things is smoother than any one of them, so a violation of that inequality means the surgery did something other than average.

### The head that shipped wrong once

The stem is half the surgery. The other half:

```python
# model/stem_swap.py:37
def reshape_detect_head(model: YOLO, nc: int) -> YOLO:
```

This used to be a no-op that only logged an intent to change `nc`, on the assumption that Ultralytics' trainer would re-initialise the head on first `train()`. It does not. Exporting straight from the swapped checkpoint produced an ONNX graph with an 80-class COCO head, so the runtime argmax ranged over 80 logits and every detection resolved to "unknown" against the 4-entry class map.

Two things now prevent that shipping again. `verify_stem` checks both ends of the surgery (`model/stem_swap.py:232`), stem in-channels *and* head out-channels, because checking only the stem is exactly what let the 80-class head through. And `postprocess` refuses to run against a mismatched head at all:

```python
# inference/engine.py:306
n_classes = pred.shape[0] - 4
if n_classes != len(CLASS_NAMES):
    raise ValueError(...)
```

The head rebuild carries over a slice of the pretrained classification bias rather than starting fresh (`model/stem_swap.py:76`). YOLOv8 initialises that bias so initial predicted objectness is low; a zeroed head starts at 0.5 probability everywhere and floods the loss with false positives for the first epochs.

### Two-phase training, and the freeze that pins the wrong layer

Phase 1 trains the stem, neck and head with the backbone frozen. Phase 2 unfreezes everything at a lower learning rate. The reasoning is that the stem and head are the two surgically modified parts, and letting their large early gradients flow into pretrained backbone features before they are calibrated is what wrecks the feature pyramid.

There is a contradiction in the repository here that matters:

```python
# model/stem_swap.py:211  (freeze_backbone)
requires = i > freeze_until          # freeze_until=9, so layer 0 is FROZEN

# model/train_6ch.py:332  (set_trainable)
trainable = True if not freeze_backbone else (i == 0 or i > 9)   # layer 0 TRAINABLE
```

`freeze_backbone` freezes layer 0, which is the newly swapped 6-channel stem: the one layer that most needs to learn. `train_6ch.py` keeps it trainable and says so at `model/train_6ch.py:34`. The trainer uses `set_trainable`, so the shipping model is trained correctly. `freeze_backbone` is a leftover that would silently do the wrong thing to anyone who called it.

Everything else in `model/train_6ch.py` is there for a measured reason worth naming briefly:

- **Preprocessing parity is enforced by construction.** Tiles enter training as float32 in [0,1], stretched to 640 square, byte-for-byte what `inference/engine.py:preprocess` does. This is why the file does not use `YOLO(...).train()`: Ultralytics' loader reads through OpenCV and divides by 255, and a detector trained on a different normalisation than it is served with is the classic way to score well offline and detect nothing on orbit.
- **Augmentation is band-agnostic on purpose** (`model/train_6ch.py:72`). HSV jitter has no meaning across B2 to B12, and mosaic would splice tiles with unrelated shorelines into one frame, teaching land-water boundaries that cannot occur. Scale jitter *is* band-agnostic and was missing, which was costing real accuracy on small objects.
- **Augmentation randomness is per-item, not per-dataset** (`model/train_6ch.py:91`). A single `random.Random(seed)` on the dataset is correct at `num_workers=0` and silently wrong above it: forked workers inherit identical generator state, and with non-persistent workers the run replays one epoch's worth of augmentation forever. Invisible locally, and it bites only on the multi-worker GPU run that can least afford it.
- **Scale-crop augmentation applies the same `min_visible` rule the tiler applies** (`model/train_6ch.py:140`). Augmentation must not be able to manufacture a label from a sliver that tiling would have discarded.
- **Checkpoint selection scores the EMA weights, not the live ones** (`model/train_6ch.py:767`), because the averaged weights are what ship. Validating one model and saving another selects an epoch on a model that is not the one written to disk.
- **The final number is scored on the full split, not the selection subset** (`model/train_6ch.py:839`). The selected epoch is by construction the subset's luckiest, and reporting that figure would bake the selection bias into the README.

### Export, and why the batch axis is static

```python
# satellite_export.py:108
def export_fp32(weights: str, out_path: Path, dynamic_batch: bool = False) -> Path:
```

Static batch of 1, for a correctness reason before a deployment one. ONNX Runtime's symbolic shape inference cannot fully resolve YOLOv8's head with a symbolic batch dimension, so `quant_pre_process` raises "Incomplete symbolic shape inference" and static INT8 quantization cannot run at all. The deployment reason is secondary and also true: the target processes one tile per attitude-stable window, and a fixed shape lets the runtime pre-plan allocations, which supports determinism. `--dynamic-batch` is available for ground-side batch evaluation and refuses to continue into quantization (`satellite_export.py:247`).

### Static INT8 post-training quantization, mechanically

```python
# satellite_export.py:181
quantize_static(
    model_input=str(prepped),
    model_output=str(int8_path),
    calibration_data_reader=reader,
    quant_format=QuantFormat.QDQ,
    activation_type=QuantType.QUInt8,
    weight_type=QuantType.QInt8,
    per_channel=True,
)
```

Four choices in seven lines. Take them one at a time.

**What a QDQ node is.** Quantization does not rewrite a convolution into an integer convolution in the graph you export. It inserts a *pair* of nodes around every tensor you want quantised: a `QuantizeLinear` that maps float to int using a scale and a zero point, and a `DequantizeLinear` that maps back. The graph stays semantically float. The optimisation happens later, in the runtime, which recognises the Q/DQ sandwich and fuses it into a genuine integer kernel:

```
   before                          after export                      at runtime
                                                                (fused by ORT)
  ┌───────┐                  ┌───┐  ┌────┐  ┌───────┐  ┌───┐    ┌──────────────┐
  │ Conv  │      ────►       │ Q │─►│ DQ │─►│ Conv  │─►│ Q │──► │ QLinearConv  │
  │(float)│                  └───┘  └────┘  │(float)│  └───┘    │  (int8 math) │
  └───────┘                    ▲             └───────┘           └──────────────┘
                        scale, zero_point         ▲
                        from calibration    weights also Q/DQ wrapped,
                                            per output channel
```

Why bother with the indirection? Because it makes the quantised model a *portable* artifact rather than a runtime-specific one. A QDQ graph runs correctly on any ONNX runtime, in float, exactly reproducing the quantised numerics; a runtime that knows the pattern gets the speed. You get a single artifact whose numerical behaviour is defined by the graph rather than by which kernels happen to be available.

**Why per-channel weights.** A convolution layer's output channels have wildly different weight ranges. Quantise the whole tensor with one scale and the channel with the largest range sets the step size for every other channel, so a narrow-range channel gets a handful of distinct levels out of 256 and its filter becomes nearly constant. Per-channel gives each output channel its own scale, which costs one float per channel in the model file (invisible against 3.69 MB) and preserves the small-magnitude filters that small-object detection depends on. Opset 13 is the minimum that supports it, which is why `satellite_export.py:149` pins it.

**Why asymmetric uint8 activations and symmetric int8 weights.** Weights are roughly zero-centred, so a symmetric int8 range with zero point 0 wastes nothing and makes the accumulation cheaper. Activations after SiLU are not zero-centred: SiLU floors at about -0.278 and rises unbounded, so the distribution is strongly one-sided. A symmetric range would spend half its 256 levels on values that never occur. Asymmetric uint8 fits the range where the data actually is, at the cost of carrying a non-zero zero point through the arithmetic.

**Why the calibration set is 32 real tiles through the identical preprocessing path.** Static quantization needs to know each activation tensor's range in advance, and it learns that by running real inputs through the float graph and recording what it sees. Two ways to get this wrong. Calibrate on random noise and the observed ranges are far wider than reality, so the int8 steps are coarse across a range the network never visits and small-object recall degrades measurably. Calibrate through a *different* preprocessing path than inference uses, and the learned scales describe a distribution that never occurs in production. `TileCalibrationReader.get_next` (`satellite_export.py:84`) therefore mirrors `inference/engine.py:preprocess` line for line, including the per-band resize and the exact interpolation flag, and the docstring says so.

32 samples is the point of diminishing returns for a network this small (`satellite_export.py:49`). It is also the source of a real, small optimistic bias, stated in the README and repeated here because it is the kind of thing that gets lost: **those 32 tiles are drawn from `val/images`, the same 3,677-tile split the INT8 accuracy is scored on.** Roughly 0.9% of the scoring set was seen by the calibrator. Calibration fits activation ranges, not weights, so the effect is second-order, but it is not zero and the INT8 column is not independent of its calibration data.

**Why static and not dynamic.** Dynamic quantization computes activation ranges at runtime, per inference. Two consequences. Most ONNX Runtime kernels leave convolutions in float under dynamic quantization, so a conv-dominated detector barely shrinks. And per-inference latency becomes data-dependent, because the range computation depends on the values. That second one is fatal here, not merely inconvenient: `AssuranceProfile.deterministic_execution_required` is `True` for both profiles, `model/benchmark_quantization.py:79` asserts that the same input produces bitwise-identical output across runs, and the whole ground-side reproduction argument rests on it. Dynamic quantization would trade the assurance property for a smaller diff.

### What quantization actually cost

| Metric | FP32 | INT8 | |
| :--- | ---: | ---: | :--- |
| Artifact size | 12.67 MB | **3.69 MB** | 3.43x smaller |
| Latency, CPU, 640 square, sequential | 94.1 ms | 51.4 ms | 1.83x faster |
| Mean relative divergence | n/a | 2.107 % | max 2.812 % |
| mAP@0.5, 3,677 real tiles | 0.889 | **0.880** | costs 0.9 points |
| mAP@0.5:0.95 | 0.576 | **0.544** | costs 3.2 points |
| Bitwise determinism | n/a | PASS | identical across runs |

`MEASURED`, `model/artifacts/quant_benchmark.json` and `accuracy_{fp32,int8}.json`. Regenerate with `python model/benchmark_quantization.py --platform skyroot-oam` and `python model/evaluate_detector.py --onnx model/artifacts/osp_yolov8n_int8.onnx --images val/images --labels val/labels`.

Two notes the table cannot carry.

The benchmark prints a warning at `model/benchmark_quantization.py:143` that deserves repeating: relative divergence characterises the quantization step in tensor space and **is not a detection-accuracy figure**. A graph can diverge by 2% and still lose every box. That is precisely why `model/evaluate_detector.py` exists as a separate script, and why the 0.9 and 3.2 point rows come from it rather than from the divergence column.

And the size target is missed. The README once advertised "<3 MB (INT8)". The measured artifact is 3.69 MB. It is recorded as missed rather than quietly restated.

Per class, the composite hides something (`MEASURED`, `model/artifacts/accuracy_int8.json`):

| class | instances | AP@0.5 | precision | recall |
| :--- | ---: | ---: | ---: | ---: |
| ship | 19,651 | 0.952 | 0.930 | 0.912 |
| airplane | 5,464 | 0.930 | 0.944 | 0.870 |
| harbor | 4,626 | 0.844 | 0.842 | 0.815 |
| storage-tank | 5,177 | 0.794 | 0.936 | **0.653** |

Storage-tank precision is 0.936 and recall is 0.653: 3,610 predictions against 5,177 ground-truth instances. The model is not confusing storage tanks with something else, it is failing to find them. Circular tanks in the old synthetic corpus were trivially separable; real ones sit in refinery clutter at varying scale, and the oriented-quad to axis-aligned-box conversion (mean 1.73x area inflation, recorded in `prep_manifest.json` by `data/dota_prep.py:381`) hurts tightly-packed tank farms more than isolated aircraft. That class is the honest weak point of this detector, and a single mAP number would have hidden it behind three healthy ones.

> ### The interview answer
>
> "The perception stack is a YOLOv8n with two pieces of surgery and one compression step. The stem goes from three channels to six, and the three new input channels are initialised to the mean of the pretrained RGB weights rather than to noise, so from step zero the network has working edge detectors on bands it has never seen. The head goes from eighty COCO classes to four, carrying over the pretrained bias calibration so it does not start predicting everything at fifty percent. Then static INT8 post-training quantization, QDQ format, per-channel symmetric weights because output channels have very different ranges, asymmetric uint8 activations because post-SiLU tensors are not zero centred, calibrated on thirty-two real tiles pushed through the exact same preprocessing path inference uses. Static rather than dynamic specifically because dynamic makes latency data-dependent, and the whole assurance story rests on bitwise determinism. It costs 0.9 points of mAP at IoU 0.5 and buys 3.4x on size and 1.8x on latency. The honest caveat is that thirty-two calibration tiles come from the same split I score on, so about one percent of the scoring set was seen by the calibrator."

**Concepts you now own.** *Post-training quantization and calibration-set design.* The transferable ideas: a QDQ graph is a portable description of quantised numerics rather than a runtime artifact; per-channel scales exist because per-tensor scales are set by your widest channel; a calibration set must come from the deployment distribution *through the deployment preprocessing*; and static beats dynamic whenever determinism is a requirement rather than a nicety. Also *training-serving skew*, prevented here by construction rather than by monitoring. These are *DMLS* chapter 7 (model compression and optimisation) and chapter 6's warning about train-serve inconsistency, and they transfer directly to any on-device or latency-budgeted model deployment.

---

## 4. Pixels to coordinates

A harbour box and the ships moored inside it overlap by construction. Run one global NMS pass over a harbour scene and you get one box: the harbour. Every vessel in it is suppressed, silently, as a duplicate.

That is not a hypothetical. It is the scene type this system exists for, and it is what class-agnostic NMS does to it.

```python
# inference/engine.py:batched_nms
keep = cv2.dnn.NMSBoxesBatched(xywh, scores, cls_ids, 0.0, iou_thresh)
```

The fix is per-class suppression: boxes of different classes never suppress each other, so a harbour cannot delete the ships inside it. This used to be a hand-written IoU loop plus the coordinate-offset trick that shifts each class into its own disjoint region of coordinate space. Both are gone. OpenCV ships the same operation as `cv2.dnn.NMSBoxesBatched`, and cv2 is already a hard dependency of the engine for tile decode and resize, so the hand-rolled version bought nothing but a second implementation to keep correct. It was checked against the replacement over 300 randomised box sets before removal.

The corpus is built to exercise this. `data/synth_demo.py:35` generates a `port` archetype specifically because it produces vessels berthed *inside* harbour boxes, which is the overlap that class-agnostic NMS destroys.

### Stretch, not letterbox

```python
# inference/engine.py:184
def preprocess(tile: np.ndarray) -> np.ndarray:
```

The conventional detector preprocessing is letterbox: scale the longest side to 640, pad the short side, remember the padding, and subtract it back out of every predicted box. OSP stretches instead, resizing each band anisotropically to 640 square.

The reason is downstream, in the geo projection:

```python
# inference/engine.py:348
lat = footprint["lat_max"] - (cy_px / tile_size) * (footprint["lat_max"] - footprint["lat_min"])
lon = footprint["lon_min"] + (cx_px / tile_size) * (footprint["lon_max"] - footprint["lon_min"])
```

That is a clean linear map from pixel space to the tile footprint, and it stays clean exactly as long as pixel `(0,0)` is the footprint's northwest corner and pixel `(640,640)` is its southeast corner. Letterboxing breaks that: now some pixels are padding, the mapping needs a scale and an offset per axis, and every consumer of `pixel_to_latlon` needs to know which letterbox parameters produced the tile it is looking at. That is a state dependency threaded through the wire format, the dashboard, the explainability panel and the tests, all to preserve an aspect ratio that does not exist, because tiles are cut square upstream by `data/dota_prep.py:tile_origins`.

So the trade is: accept anisotropic scaling on the rare non-square input, in exchange for a projection with no hidden parameters. Given every tile in the pipeline is already square, the "rare non-square input" costs nothing today.

The linearity is what the geo tests can then check cheaply. `tests/test_orbital.py:532` asserts every detection lies inside its own tile footprint, and `tests/test_orbital.py:516` asserts each footprint spans 6.4 km north-south, which is 640 px at 10 m. Both are one-line assertions only because the mapping is one line.

### The footprint is the honest weak point

The map above is exact given the footprint. The footprint is where the aerial-versus-orbital gap lands.

`tools/generate_briefs.py:88` fixes `GSD_METRES = 10.0` and derives a 6.4 km tile from it. DOTA tiles are aerial at roughly 0.1 to 1 m ground sample distance. So a DOTA tile placed on a Sentinel-2 footprint is being told it covers 6.4 km when it covers something between 64 m and 640 m: wrong by one to two orders of magnitude.

The repository handles this with a refusal rather than a footnote:

```python
# tools/generate_briefs.py:103
def check_gsd_provenance(tiles_dir: Path, allow_aerial_gsd: bool) -> str | None:
```

It looks for `prep_manifest.json`, sees `generator: data/dota_prep.py`, and **exits** unless `--allow-aerial-gsd` is passed explicitly. With the flag, every brief's `provenance.geolocation` is rewritten from "real" to "approximate", with the reason spelled out in the brief itself. You can read this in any committed brief today: `data/briefs/P0007_00000_01497.json` says `"geolocation": "approximate ... Do not treat as measured geolocation."`

The design principle worth extracting: the caveat lives in the *artifact*, not in the documentation. A reader of one brief file, with no access to this document, still cannot mistake the footprint for a measurement. `data/dota_prep.py:31` makes the same choice one level up and notes there is deliberately no opt-out flag for the manifest's GSD note, because it is not user-configurable.

> ### The interview answer
>
> "Two decisions in the post-processing path. First, NMS is per class, not global. A harbour box contains the ships moored inside it by construction, so one global NMS pass deletes every vessel in a harbour scene, which is the scene type the system exists for. The implementation shifts each class into a disjoint region of coordinate space and runs one pass, so it stays single-pass. Second, the resize is a stretch, not a letterbox. That sounds wrong for a detector until you look at what it buys: pixel to lat-lon stays a clean linear map over the tile footprint with no padding offset to thread through the wire format and every consumer. Tiles are cut square upstream anyway, so the aspect ratio it would preserve does not exist. The honest gap is the footprint itself: the tiles are aerial at sub-metre GSD and the footprint math assumes Sentinel-2's ten metres, so the brief generator refuses to run against an aerial split unless you explicitly pass a flag, and then it relabels the geolocation as approximate inside every brief."

**Concepts you now own.** *The decision path is part of the model.* NMS, class mapping and the confidence threshold are not glue around the network, they are part of what you are evaluating, which is why `model/evaluate_detector.py` scores through `inference.engine.postprocess` rather than through a training-framework metric. Also *choosing representations that keep downstream transformations invertible in one line*, which is a general defence against state that leaks across module boundaries. Chapter 6 of *DMLS* makes the first point as offline evaluation of the deployed path rather than the model in isolation.

---

## 5. The orbital layer, six modules deep

Until recently this project's orbit was a picture. A perfect 51.6 degree circle in `ground/globe.py`, no Earth rotation, and a constant longitude offset applied so the track would pass over the demo scene. Three separate fictions stacked: the wrong inclination for the spacecraft depicted, no rotation beneath the orbit, and a fudge factor to make the picture come out right. Next to it sat `contact_minutes_per_orbit = 5.0` in a config file.

Nothing in the repository knew when the spacecraft could talk to the ground. Which made the central claim, a brief instead of an image, an assertion rather than a computation.

Six modules close it, each depending only on the ones above:

```
  tle.py         parse a committed CelesTrak snapshot, grade its age
      │              TLERecord.epoch, .inclination_deg, .staleness()
      ▼
  frames.py      TEME → ECEF → WGS-84 geodetic; topocentric look angles
      │              teme_to_ecef, ecef_to_geodetic, look_angles
      ▼
  stations.py    real site coordinates + an elevation mask per site
      │              HYDERABAD.ecef_km, .elevation_mask_deg
      ▼
  propagate.py   SGP4 → SubPoint / LookAngle time series
      │              Propagator.at(t), .look_from(station, t)
      ▼
  passes.py      contiguous runs above the mask, boundaries bisected
      │              ContactWindow(aos, los, max_elevation, quality)
      ▼
  downlink.py    window × link budget → byte allowance → what fits
                     DownlinkPlan(scheduled, deferred, decisions, policy_hash)
```

Every arrow is a real dependency and there are no back edges. That is what lets each layer be tested against something external rather than against the layer below it.

### A TLE is not a state vector

This is the first thing that has to be right, and it is a data-format fact rather than an implementation detail.

A two-line element set is a set of *mean* elements, fitted so that when you push them through one specific analytical theory, that theory reproduces the observed track. The theory is SGP4. Feed the same elements to a general-purpose numerical integrator, or mix them with osculating elements from another source, and you get a confidently wrong answer: the periodic terms SGP4 removes during the fit are not present in the elements to be re-added by an integrator that does not know they are missing.

So `orbital/propagate.py:8` states it plainly: SGP4 is part of the data format's definition, which is why the module wraps the reference implementation rather than rolling its own.

The parsing has one classic trap:

```python
# orbital/tle.py:73
day_of_year = float(raw[2:])
return dt.datetime(year, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(days=day_of_year - 1.0)
```

January 1st is day 1.0, not day 0.0. Drop the `- 1.0` and every prediction shifts by 24 hours while still producing a perfectly valid datetime and a perfectly plausible pass. `tests/test_orbital.py:80` pins the decode against a known instant precisely because the failure mode is a plausible answer.

The committed-snapshot decision, and its mitigation, is documented at `orbital/tle.py:6`: SGP4 along-track error grows roughly 1 to 3 km per day from the element epoch, so a pass predicted from a six-month-old element set is fiction dressed as precision. `staleness()` grades age against thresholds derived from that error growth (`fresh` under 3 days, `usable` under 14, `stale` under 45) and the dashboard warns on the grade (`ground/dashboard.py:1053`). An old TLE is allowed to produce a number. It is not allowed to produce a number that looks freshly measured.

### The frame that is not the frame you think it is

SGP4 hands you coordinates in TEME: True Equator, Mean Equinox, of date. Not J2000. Not GCRF. Not Earth-fixed.

Getting this wrong is the single most common silent error in satellite geometry code, and the reason it survives review is that it *still produces an orbit*. Confuse TEME with J2000 and your subpoint lands on the right continent. Apply GMST with the wrong sign and your pass is about the right length. You are wrong by tens of kilometres and every plot looks fine.

TEME shares its z-axis with the true equator of date, so the transformation to Earth-fixed is a single rotation about z by Greenwich Mean Sidereal Time:

```python
# orbital/frames.py:113
theta = gmst_rad(when)
c, s = math.cos(theta), math.sin(theta)
x, y, z = r_teme
return (c * x + s * y, -s * x + c * y, z)
```

You compute GMST, rotate once about z, and you are in the Earth-fixed frame. Then Bowring's method gets you from Earth-fixed to WGS-84 geodetic in five iterations (`orbital/frames.py:119`), and a rotation into the site's East-North-Up basis gets you look angles (`orbital/frames.py:170`).

Two smaller things in that last step. Elevation is measured against the *geodetic* local horizontal, the ellipsoid normal, because that is what an antenna's elevation axis physically tracks; using the geocentric direction instead misplaces the horizon by up to about 0.2 degrees at mid-latitudes. And `julian_date` refuses a naive datetime outright (`orbital/frames.py:64`), because a naive value would be read as local time and bias every subsequent prediction by the machine's UTC offset, which is another bug that produces a plausible pass.

**Why hand-written rather than Skyfield.** Two reasons, and the second is the real one. Skyfield pulls in JPL ephemeris machinery for planetary work that satellite propagation does not need, and its timescale wants a downloadable leap-second file, which is the wrong dependency profile for a project whose argument is "this fits in a constrained environment". More importantly: the frame conversion is where this class of code goes wrong, and the failure is silent. Writing it out and checking it against an independent implementation is how that bug gets found. Hiding it inside a library call is how it ships.

So Skyfield is the oracle, test-only, never imported at runtime:

```
worst slant-range disagreement : 38.5 m
worst elevation disagreement   : 0.0003 deg
worst ground-position disagree : 41.8 m
worst altitude disagreement    : 0.00 m
```

`MEASURED`, 85 samples across 24 hours of propagation, `tests/test_orbital.py:165`. The test bounds are 50 m of ground position and 0.01 degrees of elevation: loose enough that documented omissions pass, tight enough that a genuine frame error, which produces kilometres and degrees, cannot.

The residual is exactly what `orbital/frames.py:32` names as deliberately omitted: UT1 minus UTC (bounded at 0.9 s by the definition of the leap second, about 0.4 km of Earth rotation at the equator), polar motion (under 1 arcsecond, about 30 m at the surface), and atmospheric refraction at the horizon (a second or two of rise time at a 10 degree mask). All sit far below the error already introduced by TLE age, so modelling them would be arithmetic that looks more careful without being more correct. Naming them in the module puts the omission on record as a decision rather than an oversight.

### The elevation mask is an honesty parameter

A contact window is not "when the spacecraft is over the horizon". It is when the spacecraft is high enough that the link closes. At low elevation the slant range is long, the signal cuts through far more atmosphere, and terrain intrudes. Real S-band stations work to 5 to 10 degrees.

Dropping the mask from 10 degrees to 0 would roughly double the apparent contact time for a LEO pass, which would roughly double **every downlink number in this project**, for free, by counting time when the link cannot close. `orbital/stations.py:6` says exactly this, the mask is stated per station, and `tests/test_orbital.py:236` asserts that relaxing it lengthens a given pass and can only ever reveal more passes, never fewer.

That test has a subtlety worth reading, because it is a small lesson in comparing the right things. You cannot compare the first window at 10 degrees against the first window at 5 degrees: at 5 degrees an additional earlier grazing pass appears that never clears 10 degrees at all, so the lists are not aligned. The test matches passes by peak-elevation time first, then compares. The extra pass becomes the second assertion rather than a confusing failure.

### Bisection, and why 10 seconds of grid error is a byte problem

Elevation above a station is smooth and single-peaked during a pass. So the algorithm is: sample on a fixed 10 second grid, find each contiguous run above the mask, then refine the two boundary crossings.

Refinement is not cosmetic. The grid step is the dominant error in the reported *duration*, and duration is multiplied directly by the link rate to produce a byte budget. Two boundaries each uncertain by up to 10 seconds means up to 20 seconds of a 609 second window, which at 32 kbps derated by 0.80 is:

```
20 s x 32,000 bits/s / 8 x 0.80 = 64,000 B  ≈  155 briefs
```

`DERIVED` from the measured window. That is the difference between planning 155 observations and not. Twenty bisection iterations halve a 10 second bracket to about 10 microseconds (`orbital/passes.py:49`), and each iteration is one SGP4 call, so the whole refinement costs under a millisecond. `tests/test_orbital.py:224` asserts the returned AOS and LOS sit on the mask to 1e-3 degrees.

Two more decisions in `find_passes` that keep the contact-count honest:

**Truncated windows are flagged, not extrapolated.** A run touching either end of the search span is a pass already under way at the start, or still under way at the end. Its true AOS or LOS lies outside what was propagated, so the sample time is kept and the window is marked `truncated_aos` / `truncated_los` (`orbital/passes.py:190`). `next_pass` then skips them entirely (`orbital/passes.py:233`), because planning a downlink against the tail of a pass the spacecraft is already flying through would promise a full window's bytes when only part remains. The dashboard excludes truncated windows from its passes-per-day count for the same reason (`ground/dashboard.py:605`).

**Passes shorter than 30 seconds are discarded** (`orbital/passes.py:140`). A 20 second grazing contact cannot complete antenna acquisition, let alone move data, so counting it would inflate the contacts-per-day figure with windows nobody could use.

### The two assertions that pin the track to reality

Anyone can draw a plausible orbit. Two assertions distinguish a propagated track from a drawing, and they are the reason `ground/globe.py` was rebuilt rather than patched.

```python
# tests/test_orbital.py:543
expected = 180.0 - rec.inclination_deg     # 98.57 deg  →  81.43 deg
assert max(lats) == pytest.approx(expected, abs=0.5)
assert min(lats) == pytest.approx(-expected, abs=0.5)
```

A retrograde orbit at inclination *i* reaches maximum latitude 180 minus *i*. Sentinel-2C's 98.57 degrees gives plus or minus 81.43 degrees. The old synthetic track used 51.6 degrees and would top out near plus or minus 51.6. **This one assertion fails on any track that is not built from the real element set**, and it fails loudly, by 30 degrees.

```python
# tests/test_orbital.py:560
crossings = [lons[i] for i in range(1, len(lats)) if lats[i - 1] < 0 <= lats[i]]
drift = abs(crossings[1] - crossings[0]); drift = min(drift, 360 - drift)
assert 20.0 < drift < 30.0
```

Successive ascending equator crossings must drift about 25 degrees west, because the Earth turns roughly 25 degrees during one 100.6 minute orbit. The replaced synthetic track was a **closed loop**: it returned to the same longitude every orbit, so its drift was 0. That is the second assertion, and it catches exactly the failure the first one might miss, a track with the right inclination and no Earth rotation.

A third test cross-checks two independently written pieces of geometry against each other: at peak elevation the subpoint must lie inside the station's visibility circle (`tests/test_orbital.py:599`), where the circle is computed from a closed form, `arccos(Re/(Re+h) · cos e) − e`, and the peak comes from the pass finder. If the pass finder and the globe disagreed, one of them would be wrong, and neither would say so on its own.

### The exact second matters

One more consequence of using real geometry: it constrains things you would otherwise be free to make up.

The committed brief corpus is anchored to a specific instant, `CAMPAIGN_START_UTC = 2026-08-21 05:46:00Z` (`tools/generate_briefs.py:83`), and each of the 20 tiles is placed at the subpoint the spacecraft actually occupied, spaced by the *computed* subpoint ground speed (6.7497 km/s, `MEASURED`, recorded in `data/briefs/manifest.json`) rather than by the often-quoted 7.5 km/s orbital speed, which would space tiles about 13% too far apart and leave gaps in a strip described as contiguous.

The anchor was first set at 05:44:00Z. That pass is still over inland Maharashtra: a strip labelled "Laccadive Sea" would have been over farmland. By 05:46:00Z the subpoint is at 10.8°N 72.7°E, in open water, and the whole 19 second strip stays over sea. `test_corpus_is_over_water_in_the_laccadive_sea` (`tests/test_orbital.py:524`) now enforces it, alongside `test_corpus_footprints_sit_on_the_real_ground_track` (`tests/test_orbital.py:500`), which asserts every footprint centre matches the propagated subpoint at that brief's own timestamp to 1e-4 degrees.

Two minutes of anchor is the difference between a demo that is honest and one that is decorative, and the only reason it was catchable is that the geometry was real enough to be wrong.

The corpus imagery is committed at 384 px JPEG rather than full-resolution PNG (`tools/generate_briefs.py:204`), which took 16 MB to 588 KB. Boxes are drawn at full resolution so they land on the exact detected pixels, then the composite is downscaled once. The imagery is context, not evidence: the brief JSON carries bbox coordinates at full precision, and detail no reader can use is not worth 16 MB of repository weight.

> ### The interview answer
>
> "The orbital layer is six modules, each depending only on the one above, and the whole point is that the contact window is computed rather than assumed. The part I would want a reviewer to look at is the frame chain. SGP4 hands you coordinates in TEME, which is not J2000 and not Earth-fixed, and if you get that wrong you do not get an error, you get a subpoint on the right continent that is thirty kilometres off and a pass of roughly the right length. So I wrote the conversion out explicitly, TEME to ECEF by one rotation through GMST, then Bowring to WGS-84 geodetic, then a rotation into the site's east-north-up basis for look angles, and I check it against Skyfield as a test-only oracle over twenty-four hours of propagation. Worst disagreement is 38.5 metres of slant range and three ten-thousandths of a degree in elevation, and the residual is exactly the terms I documented as omitted. Two things make the geometry honest rather than merely plausible: a ten degree elevation mask, because zero degrees would double every downlink number in the project for free, and bisection on the mask crossings, because ten seconds of grid error on each boundary is sixty-four kilobytes of the budget, which is a hundred and fifty-five briefs."

**Concepts you now own.** *Differential testing against an independent oracle*, which is the right tool whenever a component's failure mode is a plausible wrong answer rather than an exception. *The data format defines the algorithm*: a TLE is only meaningful through SGP4, the same way a compressed feature is only meaningful through the encoder that produced it. And *resolution errors compound into resource decisions*, which is the general form of the bisection argument: an error bar on a measurement becomes an error bar on everything computed from it. *DMLS* chapter 9's argument for testing in production against a reference implementation is the same instinct applied to a live system.

---

## 6. The authority boundary

Start with the failure mode, because it is the entire reason this section exists.

A language model that fails does not crash. It does not return an error code. It does not time out. It returns a well-formed paragraph, fluently written, correctly structured, containing a coordinate 400 km from anything real, a confidence value nobody measured, and a citation to a document that does not exist. Every downstream system accepts it happily, because it is valid in every way a system can check cheaply.

Now put that in a control loop on a vehicle that performs propulsive manoeuvres.

The usual architecture, ask the model what to do and then do it, buries the authority handoff inside a prompt. There is no line of code where authority transfers, so there is no line of code a reviewer can point at. That is the thing this section is designed to fix.

### Four mechanisms, in order of how hard they are to undo

**1. The policy engine decides, deterministically, from the payload alone.**

`PolicyEngine.compute_alert_level` (`agent/mission_controller.py:114`) is a pure function of the detection payload and a history count. Four named risk zones as lat-lon boxes (`agent/mission_controller.py:94`), a cluster threshold of 3, and confidence bands. No model involved. Auditable, reproducible, testable, and readable in one screen, which is deliberate: the policy is the artifact a reviewer should argue with.

**2. Reconciliation takes the higher severity, always.**

```python
# agent/mission_controller.py:358
def _reconcile_alerts(self, policy: str, llm: str) -> str:
    order = {"GREEN": 0, "YELLOW": 1, "ORANGE": 2, "RED": 3, "UNKNOWN": 0}
    return policy if order.get(policy, 0) >= order.get(llm, 0) else llm
```

Three lines, and the asymmetry is the whole design. Read it as a state machine:

```
                          LLM says
                GREEN   YELLOW   ORANGE   RED
              ┌───────┬────────┬────────┬───────┐
   P    GREEN │ GREEN │ YELLOW │ ORANGE │  RED  │  ← LLM may escalate
   o          ├───────┼────────┼────────┼───────┤
   l   YELLOW │YELLOW │ YELLOW │ ORANGE │  RED  │
   i          ├───────┼────────┼────────┼───────┤
   c   ORANGE │ORANGE │ ORANGE │ ORANGE │  RED  │
   y          ├───────┼────────┼────────┼───────┤
       RED    │  RED  │  RED   │  RED   │  RED  │  ← LLM can never de-escalate
              └───────┴────────┴────────┴───────┘
                  ▲
        unparseable / missing / UNKNOWN maps to 0,
        so a broken model is exactly as harmless as a silent one
```

The model can raise an alarm and can never lower one. And `UNKNOWN` maps to 0, so the three ways the LLM can fail (unavailable, timed out, unparseable) collapse into the one behaviour: the policy verdict stands and the pipeline continues. `_build_policy_fallback_brief` (`agent/mission_controller.py:413`) fills a structurally valid brief carrying the policy verdict, so no downstream consumer needs to branch on whether the model ran.

**3. The scheduler's interface has no seam.**

```python
# orbital/downlink.py:519
def plan(self, window: ContactWindow, candidates: Iterable[BriefCandidate]) -> DownlinkPlan:
```

A window and a list of briefs. There is no argument, hook or callback through which a model can reach the decision: not an optional one, not an ignored-by-default one. Priorities come from `score_brief` (`orbital/downlink.py:287`), a pure function of a brief's own fields. `DownlinkPlan` is frozen (`orbital/downlink.py:359`), so the object handed to the analyst for narration cannot be edited by it.

That last point is the difference between a convention and a property. A convention says "the analyst should not modify the plan". A frozen dataclass says attempting it raises `FrozenInstanceError`, which is what `tests/test_orbital.py:431` asserts.

Two choices inside `plan()` follow from the same reasoning and both cost measurable quality on purpose.

**Greedy first-fit, not optimal packing.** This is technically a knapsack problem and greedy is not optimal for it, so the plan leaves a few percent of window utilisation on the table. Optimal packing improves utilisation by *reordering*: dropping a higher-priority observation to slot in a smaller lower-priority one. Priority order is the mission. Utilisation is a diagnostic. Trading the first for the second produces a worse system that scores better, so strict priority order is the correct behaviour rather than a shortcut.

**A hand-written scoring function, not a learned ranker.** A learned model would almost certainly order briefs better. It would also make the answer to "why was this brief dropped?" a matrix multiplication. On a vehicle where every autonomous action must trace to a rule, that is a worse system even when it ranks better. The policy is six named constants readable in one screen (`orbital/downlink.py:65` through `:93`), deliberately, because the policy is the artifact a reviewer should argue with.

**4. The decision carries the policy that produced it.**

```python
# orbital/downlink.py:466
def policy_fingerprint() -> str:
```

A SHA-256 over the six scheduling constants, truncated to 12 hex characters, stamped into every plan. Today it is `5fbc14d78913` (`MEASURED`). Change `CLASS_WEIGHT["ship"]` and it changes, which `tests/test_orbital.py:400` asserts by monkeypatching the constant.

This is the difference between an audit trail and a log file. A log file says what happened. An audit trail lets you re-derive whether what happened was correct under the rules in force at the time. Without the hash, a plan from last month replayed against this month's constants produces a different answer and nothing anywhere says the comparison was invalid. With it, the mismatch is visible immediately.

Every decision also carries its rule, one of three (`orbital/downlink.py:567`, `:585`, `:596`): `oversize-brief`, `fits-in-budget`, `budget-exhausted`. The dashboard renders every one of them with the running byte count (`ground/dashboard.py:700`).

### The three tests that hold the line

Architecture claims that are not tested are decoration. Three tests make these executable.

```python
# tests/test_orbital.py:416
def test_scheduler_interface_exposes_no_model_hook():
    params = set(inspect.signature(DownlinkScheduler.plan).parameters)
    assert params == {"self", "window", "candidates"}
    ctor = set(inspect.signature(DownlinkScheduler.__init__).parameters)
    assert ctor == {"self", "downlink_kbps", "max_payload_bytes", "efficiency"}
```

**What it catches.** Someone adds `advisor=None`, or `hint=None`, or `llm_client=None` to `plan()`. That parameter *is* the authority handoff, even if the first version of the code ignores it, because the next change makes it live and the review that would have caught it has already happened. Equality against a literal set rather than a subset check is deliberate: an added parameter fails, a removed one fails, and a renamed one fails. The failure message points at the exact review question.

```python
# tests/test_orbital.py:441
a = sched.plan(w, cands).to_dict()
b = sched.plan(w, list(reversed(cands))).to_dict()
assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
```

**What it catches.** Any dependency of the plan on input order, iteration order, dict ordering, floating-point accumulation order, or hidden state carried between runs. Reversing the input is the cheap way to catch all of them at once. This is the property that makes a decision replayable, which is what the policy hash is for. It passes because ties break on `scene_id` (`orbital/downlink.py:546`), not on arrival order.

```python
# tests/test_orbital.py:458
plan = sched.plan(w, cands); before = json.dumps(plan.to_dict(), sort_keys=True)
narrative = fake_analyst(plan)
assert json.dumps(plan.to_dict(), sort_keys=True) == before
```

**What it catches.** The end-to-end version of the claim. A narrator that mutates a list it was handed, a `to_dict` that memoises and returns a modified view, a plan that lazily computes something on first read. The stand-in analyst reads everything a real one would be given, including `decisions` and `utilisation`, so the test covers the read surface an LLM would actually touch.

### Where the boundary is softer than it looks

This is the part of the project a reviewer should press on, so it is stated before they have to find it.

```python
# agent/mission_controller.py:394  (the preceding comment calls this "advisory")
llm_ovv = llm_brief.get("ovv_recommendation", {})
if llm_ovv.get("trigger") and llm_ovv.get("target_coords"):
    tc  = llm_ovv["target_coords"]
    key = (round(tc[0], 2), round(tc[1], 2))
    if key not in seen_coords:
        seen_coords.add(key)
        ovv_requests.append(OVVRequest(
            request_id    = f"OVV-{len(ovv_requests)+1:03d}",
            target_coords = tc,
            reason        = llm_ovv.get("reason", "LLM recommendation"),
            priority      = llm_ovv.get("priority", 3),
            source        = "llm",
            confidence    = 0.0,
        ))
```

**`_decide_ovv` accepts an LLM-sourced tasking request when policy has not already covered that coordinate.** This is the one place in the repository where a model's output reaches a field named "action". Be precise about what authority that is and is not.

**What it is not.** It is not a downlink decision: `DownlinkScheduler.plan` never sees an `OVVRequest`, and no code path connects them. It is not an alert level: that went through `_reconcile_alerts`, where the model cannot de-escalate. It is not an uplinked command: in this repository an `OVVRequest` becomes a line in a mission log (`agent/mission_controller.py:517`) and a card in the dashboard. Nothing transmits it, and no spacecraft acts on it.

**What it is.** It is a model-authored proposal, tagged `source="llm"`, that enters the same list as policy-authored proposals and is presented to an operator as an OVV request. On a system where OVV requests were actually uplinked, this line would be the thing to remove. And there is a second-order effect that is easy to miss:

```python
# agent/mission_controller.py:410
ovv_requests = sorted(ovv_requests, key=lambda r: r.priority)[:3]
```

The cap is three, sorted by a priority number that, for the LLM entry, **the LLM supplied itself** (`llm_ovv.get("priority", 3)`). Python's sort is stable and policy requests are appended first, so ties keep policy ahead. But an LLM claiming priority 1 sorts *above* a policy request at priority 2 or 3, and if there are more than three requests it displaces one. So the model cannot create a policy request and cannot suppress one directly, but it can outbid one for a limited slot by asserting its own urgency. That is a genuine, if narrow, channel, and the fix is one line: sort by `(source != "policy", priority)`.

Both facts are worth holding together. Four mechanisms make authority a property of the interface, and one function accepts a model-authored proposal into an action list where it can outbid a rule-authored one for a capped slot. The first four are the architecture. The fifth is where the architecture is not yet finished.

`config/platforms.py:92` states `llm_in_control_loop = False` as a profile constraint, a property of the deployment target rather than a convention someone can refactor away, and `tests/test_resilience.py:395` asserts it for every profile while also checking structurally that no fallback handler has a seam for an advisor.

> ### The interview answer
>
> "A language model's failure mode is fluent, confident, well-formed nonsense. It does not crash and it does not return an error code, which makes it uniquely dangerous anywhere downstream systems accept structurally valid input. So I made the authority boundary a property of the interface rather than a convention. Alert levels are computed deterministically from the payload by a policy engine with no model in it; the LLM produces its own assessment; reconciliation takes the higher severity, so the model can escalate and can never de-escalate, and unparseable output maps to the lowest level, which means a broken model behaves exactly like an absent one. The downlink scheduler's plan method takes a window and a list of briefs, and there is no parameter through which a model could reach the decision, which a test asserts by inspecting the signature. The plan object is frozen, so the analyst that narrates it physically cannot edit it, and every plan carries a hash of the policy constants, which is what makes it an audit trail rather than a log. Where it is softer than it looks: the OVV decision function does accept an LLM-proposed re-image target when policy has not already covered that coordinate, and because the list is capped at three sorted by priority, and the model supplies its own priority number, it can outbid a policy request for the last slot. That request never reaches the scheduler and never gets uplinked in this repo, but it is the one place a model output lands in an action list, and I would fix the sort before it ever did."

**Concepts you now own.** *Deterministic policy layers around stochastic components*, and the specific technique of making a boundary structural rather than procedural: frozen result types, signature assertions in tests, and asymmetric reconciliation where the safe direction is free and the unsafe direction is impossible. Also *policy fingerprinting* as the difference between an audit trail and a log. This is the governance and guardrails material in *DMLS* chapters 9 and 10, and the same pattern is what production systems use around any generative component that touches a workflow with consequences.

---

## 7. Failure

For most of this project's life, `config/platforms.py` said this about the `skyroot-oam` profile:

```python
watchdog_timeout_s        = 5.0
max_inference_latency_ms  = 400.0
fallback_on_model_failure = "hold_last_known_good_and_flag_ground"
```

and **none of those were reachable by any code path.**

That state is worse than having no declaration at all. It reads as a safety property and behaves as a comment with a type annotation. A reviewer sees a named fallback and assumes a fallback exists. Nobody had ever seen it execute.

### Declared behaviours, as executable code

The fix has three parts and the third is the one that matters.

First, `inference/engine.py` no longer raises. `run_tile` is a guard around `_perceive`: any exception out of the perception path, and any pass that overruns the watchdog, is converted into the profile's declared fallback brief, flagged `degraded` (`inference/engine.py:628`).

Second, each declared string resolves to a real handler in `FALLBACK_HANDLERS` (`inference/engine.py:497`). `moi-1a` gets `emit_empty_brief_with_cloud_estimate`: a well-formed brief with no detections but a live cloud estimate, because cloud cover is a threshold over one band that costs microseconds and does not touch the model, so it survives exactly the failures that take the detector down. `skyroot-oam` gets `hold_last_known_good_and_flag_ground`: the last successful detection set, re-asserted and flagged as held, because on a manoeuvring stage losing perception for one tile is not an emergency but silently losing the scene picture while the vehicle continues to act on it is.

Third, and this is the structural move:

```python
# inference/engine.py:520
declared = self.profile.assurance.fallback_on_model_failure
if declared not in FALLBACK_HANDLERS:
    raise ValueError(
        f"Platform profile '{self.profile.key}' declares "
        f"fallback_on_model_failure='{declared}', which has no handler "
        f"in FALLBACK_HANDLERS. ..."
    )
```

**An engine refuses to start against a profile whose declared fallback has no implementation.** A profile cannot promise a behaviour the code cannot perform, and the moment to discover a mis-declaration is at startup on the ground, not at the moment of the failure it was written to survive. `tests/test_resilience.py:133` constructs a profile declaring `pray_and_continue` and asserts the constructor raises.

Three smaller decisions in the fallback path are worth naming because each closes a specific way this could rot:

- **A held brief copies its anomalies rather than aliasing them** (`inference/engine.py:487`). Sharing the list means editing one payload rewrites history in the other. `tests/test_resilience.py:311` asserts non-aliasing.
- **A degraded brief never becomes the last-known-good** (`inference/engine.py:638`, only reached on a clean, in-budget pass). Otherwise repeated failures would hold a hold of a hold, and staleness would compound invisibly. `tests/test_resilience.py:322` asserts it.
- **A hold with no history degrades further rather than inventing** (`inference/engine.py:474`). A failure on the first tile of a campaign has nothing to hold, so it emits an empty flagged brief and says so in the fault string, rather than fabricating a plausible one.

And the nominal path is byte-for-byte what it was before the guard existed: a healthy brief carries none of the degradation fields (`tests/test_resilience.py:350`), so the fallback machinery costs zero bytes in the case that matters.

### The coverage test

```python
# tests/test_resilience.py:152
ASSURANCE_COVERAGE = {
    "deterministic_execution_required": "test_the_degraded_path_is_deterministic",
    "llm_in_control_loop":              "test_no_fallback_path_consults_a_model",
    "watchdog_timeout_s":               "test_a_stall_trips_the_watchdog_and_fires_the_fallback",
    "max_inference_latency_ms":         "test_a_latency_budget_breach_is_reported",
    "fallback_on_model_failure":        "test_a_model_crash_produces_the_declared_fallback",
}

def test_every_assurance_field_is_exercised():
    declared = {f.name for f in dataclasses.fields(AssuranceProfile)}
    assert declared == set(ASSURANCE_COVERAGE)
    for field_name, test_name in ASSURANCE_COVERAGE.items():
        assert test_name in globals()
```

**What it catches.** Someone adds a field to `AssuranceProfile`, say `max_memory_mb` or `require_signed_model`, ships it as a declared safety property, and never writes code that reads it. The test fails immediately with the field name. It also catches the reverse: a test named here that no longer exists, or a field renamed without renaming the mapping.

This is the same idea as `test_scheduler_interface_exposes_no_model_hook`, pointed at failure behaviour instead of at the authority boundary. Both are structural tests over a *declaration*, and both exist because the specific decay this repository suffered was declarations drifting away from code while continuing to read as guarantees. Neither test checks that a behaviour is correct. Both check that it is *connected*.

### What is caught, and what is not

```
        fault                          what the system does              caught by
  ─────────────────────────────────────────────────────────────────────────────
  model crash / EP fault        →  declared fallback, flagged degraded   guard
  perception overruns watchdog  →  same fallback, fault=WatchdogExpiry   guard
  over latency budget, returns  →  reported, brief still stands          warning
  first-tile failure, no hold   →  empty flagged brief, invents nothing  handler
  truncated/malformed brief     →  quarantined with a reason             ingest
  ─────────────────────────────────────────────────────────────────────────────
  single flipped byte in brief  →  ~half the time, NOTHING                  ✗
  bit flips in INT8 weights     →  NOTHING. Nothing at all.                 ✗
```

The watchdog is honest about what it models (`inference/engine.py:402`). On real flight hardware a watchdog is an external timer that resets the compute, and software does not get to observe its own overrun. Here the overrun is detected in-process, after the fact. What is modelled faithfully is the *recovery* path: fall back to the declared behaviour and flag ground, which is what the flight system would do on reset. What is not modelled is the reset itself.

### The first fault the model cannot see: silent bit flips, which get louder

A single-event upset is the characteristic failure of flight compute, not an exotic one, and in INT8 weights it lands as silent numerical corruption. Which bit flips matters enormously: INT8 weights are two's complement, so a flip in bit 7 changes a weight by 128 quantisation steps and a flip in bit 0 changes it by one. `flip_weight_bits` (`resilience/faults.py:82`) places flips uniformly over the weight bit population, weighting tensor selection by element count so a bit is equally likely anywhere in weight memory rather than equally likely per tensor. That is the right null model for particle strikes: the hardware has no idea which bit is the sign bit.

The model holds **25,026,816 bits** of quantised weight memory (`MEASURED`, `resilience/artifacts/degradation.json`).

Committed sweep, on the DOTA held-out split, 96 tiles per point, 3 seeds:

| bits flipped | share of weight memory | mAP@0.5 | detections emitted |
| ---: | ---: | ---: | ---: |
| 0 | 0% | 0.836 | 856 |
| 16,384 | 0.07% | 0.801 | 643 |
| 32,768 | 0.13% | 0.622 | 475 |
| 65,536 | 0.26% | 0.338 | 225 |
| 131,072 | 0.52% | 0.030 | 45 |
| 262,144 | 1.05% | 0.000 | 1 |
| 524,288 | 2.09% | 0.000 | **1,341** |
| 1,048,576 | 4.19% | 0.000 | **16,548** |

`MEASURED`, `resilience/artifacts/degradation_dota.json`, regenerate with
`python resilience/degradation.py --images val/images --labels val/labels --tiles 96`.

Read the right-hand column, and read it to the bottom. Detections fall as the model degrades, and then past 1% of weight memory they **explode**: 856 at baseline, 1 at the point of total collapse, then 1,341 and finally 16,548 at the far end, 19x the clean baseline and essentially all of it wrong. The graph loads cleanly throughout, every tensor is the right shape, inference returns normally, and nothing anywhere raises. The failure mode is not silence. It is confident nonsense, and it is not even monotone, so a naive "did we get a plausible number of detections" heuristic passes at both ends and fails in the middle.

Which is the same argument this repository makes about language models, arriving unexpectedly at the detector. A component whose failure produces well-formed output cannot be guarded by checking whether it produced output.

This is the one fault in the table that the declared fallback cannot catch, because there is no error to catch. The declared fallbacks catch failures the system can *see*.

### What catches it instead

If the perception path cannot see the fault, the check has to sit outside the perception path and never ask the model anything. That is what memory scrubbing is, and it is standard flight practice rather than an invention here: hold a golden copy of the weights in protected storage, walk working memory periodically, repair what has drifted.

`resilience/protect.py` implements the two halves separately, because a platform may afford one and not the other:

| | state needed | detects | repairs |
| :--- | :--- | :--- | :--- |
| `verify()` | 252 B, one CRC-32 per weight tensor | yes | no |
| `scrub()` | the above plus a golden copy of the weights | yes | yes |

Detection alone is nearly free — 252 bytes of state against a 3.69 MB artifact, one linear pass in **15 ms** — and it is worth having on its own. A spacecraft that can only detect still knows to stop trusting its own detections and to request an uplink, which is a great deal better than the silent degradation it replaces.

CRC-32 rather than a cryptographic hash, deliberately: the adversary is a particle, not a person. CRC-32 catches every 1-, 2- and 3-bit error in a block this size and all odd-weight errors, which is the failure mode exactly, and `zlib.crc32` is stdlib and implemented in C.

The end-to-end check runs inside the committed sweep rather than only in a test, because a safety property nobody has watched execute is a comment with a type annotation:

```
65,536 flips, spread across all 63 weight tensors
   detected  63 of 63
   mAP@0.5   0.836  ->  0.359  ->  0.836
                          bad       after scrub
```

`fully_restored: true` in `resilience/artifacts/degradation_dota.json` asserts the last two are *equal*, not close. The repaired weights are byte-identical to the golden copy, so any difference at all would mean the evaluation pipeline is non-deterministic and the entire degradation curve above is noise.

### Which bits are worth protecting

The uniform sweep is the correct physical model — a particle has no idea which bit is the sign bit — and it is the wrong measuring instrument, because it reports the average of eight populations worth wildly different amounts. Confining all 65,536 flips to one bit position at a time:

| bit flipped | weight moves by | mAP@0.5 | retained |
| ---: | ---: | ---: | ---: |
| 0 | 1 | 0.839 | 1.003 |
| 1 | 2 | 0.846 | 1.011 |
| 2 | 4 | 0.842 | 1.007 |
| 3 | 8 | 0.830 | 0.993 |
| 4 | 16 | 0.796 | 0.952 |
| 5 | 32 | 0.623 | 0.745 |
| 6 | 64 | 0.130 | 0.156 |
| 7 (sign) | -128 | **0.000** | **0.000** |

`MEASURED`, same artifact. Retention around 1.0 in the low bits is not the model improving under radiation; it is the noise floor of a 96-tile evaluation, and it is the right scale against which to read 0.156 four rows down.

The bottom half of the byte is free. The top half is where the model lives, and the cost tracks the size of the weight perturbation, which is what an INT8 two's complement weight should do and is worth having measured rather than assumed. The consequence: **a scheme protecting only the top four bits would buy essentially all the available safety for half the state.** Nothing here implements that, because a CRC over the whole tensor is cheap enough on a 3.69 MB model and a second half-built scheme is worse than one that works. It is recorded because it is the first thing to reach for if a golden copy ever stops fitting in memory.

### How often to scrub

With a degradation curve and a tolerance, the interval is arithmetic. The detector holds 95% of baseline mAP through **16,384 flips**, 0.065% of its 25,026,816 weight bits, so:

| upset rate (per bit per day) | scrub every |
| :--- | ---: |
| 1e-5 | 65.5 days |
| 1e-6 | 654.7 days |
| 1e-7 | 6,547 days |

That rate is the one number in this section tagged `ASSUMED` rather than `MEASURED`, and it cannot be anything else: OSP has no flight hardware and therefore no radiation test report. `UPSET_RATE_RANGE` in `resilience/protect.py` carries the order of magnitude commercial SRAM is generally quoted at in low Earth orbit, and everything downstream is linear in it, so a reader holding a real device's test data can rescale the table without rerunning anything.

Two ceilings on that arithmetic, stated because they both point the same way: upsets arrive as a Poisson process, so half of all intervals see more than the mean and a real mission would size against a tail quantile rather than the expectation; and multi-bit upsets, where one particle flips several adjacent cells, are ignored entirely. Both shorten the true interval. The `ponytail:` comment on `scrub_interval_hours` records the upgrade path.

**One honesty note on this table.** It is a conditional measurement, not a radiation model (`resilience/__init__.py`). It says what survives given N flips. The rate at which N flips occur is supplied separately, is `ASSUMED`, and is dealt with three subsections down.

> **Closed.** This section used to carry a second note, and it was the more embarrassing one: the committed artifact was the *synthetic-split* sweep (baseline 0.996, 119 detections) while the README quoted a DOTA re-run whose artifact had never been committed. Two documents describing different runs, and the live dashboard rendering the older of them, including a band-dropout caption the README had already corrected. The DOTA sweep is now committed as `resilience/artifacts/degradation_dota.json`, the tables above and in the README are both generated from it, and `ground/dashboard.py` prefers it over the synthetic file. The synthetic sweep stays only as a fallback for a checkout that has not regenerated anything.
>
> The re-run reproduces the README's figures exactly, which is the outcome that was in doubt: baseline **0.836**, **856** detections, **16,548** at 4.19%, all matching what this document had been quoting from an artifact nobody could check. An intermediate 24-tile run did *not* match (baseline 0.895, 165 detections, 4,002 at the far end), and that disagreement is worth recording rather than discarding: a 24-tile mAP is a noisy instrument, the sample size is part of the measurement, and the sweep is committed at 96 tiles per point precisely because that is the size the quoted numbers were taken at. The collapse-then-explosion *shape* reproduces at every sample size. The magnitudes need the sample size stated alongside them.

That had a second consequence, and it is worth recording how long it survived. The dashboard's band-dropout caption asserted that zeroing any single band, including the SWIR pair, "costs this model nothing measurable". That was the synthetic result. The README had already carried the corrected DOTA measurement, where B11 and B12 together cost 0.046 mAP, for several commits before anyone noticed that the deployed page was still stating the retracted one. A document can retract a claim; a running service keeps serving whatever artifact it was pointed at. The caption is now computed from the committed DOTA figures rather than written by hand, which is the only version of this fix that cannot go stale again.

### The second uncaught fault: one flipped byte

```python
# tests/test_resilience.py:553
def test_a_single_flipped_byte_can_survive_ingest_undetected(briefs):
    ...
    assert survived > 0, "no flipped byte survived ingest, so this corpus no longer demonstrates the gap"
    assert survived < 40, "every flipped byte survived; ingest is validating nothing"
```

**What it catches, and what it documents.** The two-sided assertion is the interesting part. `survived < 40` is a normal test: ingest must reject *something*. `survived > 0` is the opposite of a normal test. It fails if the gap ever stops existing, which keeps the limitation measured rather than hoped past.

About half the time a flipped byte lands somewhere that still parses and still type-checks, and ingest returns a brief that is well-formed and wrong. No amount of schema checking fixes this, because nothing about the payload is invalid. The real answer is an integrity check on the wire, and OSP does not have one.

What ingest *does* guarantee is narrower and worth stating precisely: **it never raises, and it never repairs.**

```python
# orbital/downlink.py:121
class BriefIngestError(ValueError):
```

A truncated brief whose `anomalies` list did not survive would coerce to "zero detections" under any best-effort parser. That is not a missing observation, it is a *false* one: the ground would record "nothing here" for a tile the spacecraft may well have found something in, and the scheduler would score it at floor priority and spend real bytes downlinking it. So the ingest layer rejects rather than guesses, and `load_brief_candidates` (`orbital/downlink.py:240`) returns both halves, the survivors and the casualties with their reasons, because losing a brief costs one observation while raising through the ground segment costs the entire contact, and a silent drop is its own failure.

`tests/test_resilience.py:611` asserts the refusal to repair directly, and `tests/test_resilience.py:584` asserts that a queue with both survivors and casualties still yields a plan.

### The mutation check

The suite was checked by mutation rather than by running it: disabling the watchdog comparison fails two tests, and letting a degraded brief become the last-known-good fails a third. That is the only evidence that a passing suite is doing work, and it is the same instinct as the `survived > 0` assertion above. A test that cannot fail is not a test, which this repository learned the expensive way (section 9).

> ### The interview answer
>
> "The platform profile declared a five second watchdog, a four hundred millisecond latency budget and a named fallback, and for most of the project's life none of them were reachable by any code path. That is worse than declaring nothing, because it reads as a safety property and behaves as a comment. So each declared string now resolves to a real handler, the engine refuses to start against a profile whose declared fallback has no implementation, and a coverage test fails if anyone adds a field to the assurance profile without adding a test that makes it happen. Then I injected each fault deliberately. Two of them nothing catches. A single flipped byte in a brief survives ingest about half the time as a well-formed wrong observation, because nothing about it is invalid, and the fix would be an integrity check on the wire that I have not built. And bit flips in INT8 weights are completely invisible: the graph loads, every tensor has the right shape, inference returns, and as the model degrades it emits *more* detections, not fewer, a hundred and nineteen at baseline against six and a half thousand at the far end, essentially all wrong, with nothing raising anywhere. Which is exactly the argument I make about language models, showing up again in the detector."

**Concepts you now own.** *Fault injection as a design tool*, not as a testing chore: you learn the shape of a failure by causing it, and the shape determines whether any guard can help. *Silent failures are the expensive class*, which is *DMLS* chapter 8's central point about data distribution shifts and monitoring: the failures that matter are the ones that do not raise. And *coverage tests over declarations*, which is a cheap, general defence against configuration drifting away from behaviour in any system where a config file reads like a guarantee.

---

## 8. The ground segment

A metric that cannot fail is not a metric. The one this ground segment shipped with was:

```python
len(payload["anomalies"]) == len(brief["anomaly_assessments"])   # → 1.0 else 0.0
```

reported as "Grounding Accuracy (Faithfulness)". It returns a perfect 1.0 when the model reports a harbor where telemetry said airplane, invents coordinates 400 km from the detection, fabricates every confidence value, cites knowledge-base IDs that do not exist, and escalates an empty ocean scene to RED. It also counted the synthetic `conf: 0.5` entries that the old regex salvage path invented on parse failure, so a *total* LLM failure could score as faithful.

Everything in `ground/eval_suite.py` is the reconstruction of that metric into six things that can each fail on their own.

### Parity first: two encodings, one schema

`payload_to_json` (`inference/serialization_utils.py:217`) does not serialise the dataclass. It converts to protobuf and back out to JSON through `_proto_to_json_str`, so the dashboard's JSON and the wire binary are guaranteed to carry the same fields by construction rather than by discipline. Any field that does not survive the proto round trip is visibly absent from both, which is how the missing `degraded` field in section 2 is discoverable at all.

### Retrieval, and three ways it silently lies

`rag/` embeds 14 curated maritime knowledge chunks (`rag/knowledge_base.py`) and ranks them per payload, injecting the top few into the prompt. Three failure modes were real and each has a fix worth naming, because all three produce *plausible* wrong retrieval rather than errors.

**The stale index.** This one is now fixed by deletion. A persisted vector index stores vectors positionally, and retrieval maps result index *i* back to `self._chunks[i]` — a mapping that is only valid if the corpus is byte-identical to the one embedded. Editing or reordering `knowledge_base.py` silently returned confidently wrong documents forever, with no error and no way to notice from the output, so the index carried a content hash over `(id, title, content)` of every chunk in order, checked on load alongside backend and count, with a rebuild on any mismatch. All of that existed to avoid re-embedding fourteen short strings. The corpus is now embedded at construction into an in-memory `(14, 384)` matrix and ranked with a dot product, which is what a flat inner-product index computes anyway. There is no persisted index left to go stale, so the fingerprint, the metadata sidecar, the integrity gate and the FAISS dependency are all gone.

**The wrong projection.** `text-embedding-004` is an *asymmetric* model: it projects documents and queries into deliberately different regions of the space. Embedding a query with `retrieval_document` means querying with a vector the index was never built to be searched by. Fixed by switching on `is_query` (`rag/retrieval.py`).

**Unnormalised vectors in an inner-product ranking.** An inner product only equals cosine similarity on unit vectors. Gemini does not return normalised vectors, so raw dot products let vector *magnitude* dominate ranking. Fixed with an explicit normalisation (`rag/retrieval.py`).

All three share a signature: retrieval still returns *k* results, ranked, plausible, and wrong. None of them raises.

### Episodic memory

`ground/scene_memory.py` is SQLite with two tables and three indexes, holding scenes and anomalies across passes. `query_region` does a bounding-box prefilter in SQL then refines with Haversine (`ground/scene_memory.py:254`), which is the right shape: the index does the cheap work and the exact distance does the correct work.

One naming imprecision worth flagging, since the LLM reads this text: `HistoricalAnomaly.pass_number` is documented as "how many orbital passes ago" (`ground/scene_memory.py:51`) but is assigned as the enumeration rank of the returned row (`ground/scene_memory.py:309`). It is a recency ordinal, not a pass count. The string reaches the prompt as "observed 3 pass(es) ago", which is a claim the field does not support.

### Six axes, and why the composite is a minimum

| Axis | Catches | Key line |
| :--- | :--- | :--- |
| `schema_validity` | structurally unusable output | `ground/eval_suite.py:141` |
| `entity_grounding` | omissions, hallucinated detections, class substitutions | `ground/eval_suite.py:261` |
| `coordinate_fidelity` | positions the model invented rather than transcribed | `ground/eval_suite.py:312` |
| `numeric_fidelity` | confidence values that were never downlinked | `ground/eval_suite.py:376` |
| `citation_validity` | fabricated sources, and real sources never retrieved | `ground/eval_suite.py:412` |
| `policy_consistency` | disagreement with the policy engine, direction-weighted | `ground/eval_suite.py:471` |

Four design decisions inside that table are the substance.

**Matching is geodesic, not positional.** The model is free to reorder, merge or drop assessments, so zipping by index would manufacture false substitutions. `match_entities` (`ground/eval_suite.py:193`) solves a greedy nearest-neighbour assignment gated at 3 km, which is order-invariant. A coordinate-free fallback matches by type only, so a model that omits coordinates is penalised on `coordinate_fidelity` rather than double-counted as a hallucination.

**Two different distance tolerances, for two different questions.** `MATCH_RADIUS_KM = 3.0` gates *which detection an assessment refers to*: a 640 px tile at 10 m spans 6.4 km, so 3 km is half a tile, generous enough to absorb rounding and tight enough that a position invented elsewhere in the scene cannot match. `COORD_FIDELITY_KM = 0.5` then asks *how faithfully it was transcribed*, once identity is settled. Using one tolerance for both would either let drift pass or split one object into a miss plus a hallucination.

**Policy deviation is scored asymmetrically.**

```python
# ground/eval_suite.py:508
score = max(0.0, 1.0 - 0.25 * delta)        # over-escalation
# ground/eval_suite.py:514
score = max(0.0, 1.0 - 0.5 * abs(delta))    # under-escalation
```

Over-escalation costs an operator's attention. Under-escalation means a real threat was downgraded by a stochastic component that the policy engine had already flagged, which is the failure a safety review actually cares about. Twice the penalty per level. Note this axis is also a second, independent check on the reconciliation logic of section 6: the policy engine wins at runtime, and this measures how often the model would have been wrong if it had not.

**The composite is a minimum.**

```python
# ground/eval_suite.py:556
composite=min(a.score for a in axes.values()),
```

A mean lets five good axes carry one catastrophic one. A brief that invents coordinates is not redeemed by having valid JSON, and in a mission-assurance context the weakest axis is the one that matters. This is the same reasoning as the per-class accuracy table in section 3: composites hide dead components, so you either report the parts or report the worst.

All six original failures are now regression cases, and the harness has a CI-safe mode that scores pre-computed briefs with no API calls (`--briefs`) alongside the live one.

The generation side was fixed too, at the decoding level rather than the prompt level. The previous version asked the model *in the prompt* never to use double quotes and to make sure its JSON was not truncated, then salvaged failures with a regex scraper that fabricated `conf: 0.5` for recovered anomalies, feeding invented numbers straight into the faithfulness metric. Now the schema is declared to the provider (`ground/llm_analyst.py:125`) so malformed JSON is not representable, and `propertyOrdering` puts `reasoning_trace` before the assessments so the model commits its reasoning tokens before it emits verdicts. On a parse failure the analyst returns `alert_level: UNKNOWN` with empty lists and a `_parse_error` (`ground/llm_analyst.py:387`), which reconciliation maps to severity 0 and `schema_validity` scores as a failure. A degraded read is fine; inventing data to make a metric pass is not.

That loose end is now closed by subtraction. The dashboard used to offer a three-way provider selector and `OrbitalAnalyst` used to branch on it, and two of the three branches had never run: `"anthropic"` pointed an OpenAI client at `api.anthropic.com/v1`, which is not an OpenAI-shaped endpoint and could not return a response, and the OpenAI path was absent from both deployment manifests, so the artifact anyone actually runs could not reach it. Gemini is the only provider now, in one code path with no selector. Provider-agnosticism you have never executed is not portability, it is an untested claim of it, and it costs a dependency and a branch to keep making.

### Explainability, and what it is not

`inference/explainability.py` computes per-band contrast between a detection's box and its surrounding background, ranks the six bands by normalised absolute contrast, and calls the top two "dominant bands".

Be careful about what that is. It is a **post-hoc statistic about the input**, not an attribution of the network. No gradient is computed, no activation is inspected, and the ranking would be identical for a detection the model got wrong. Calling it "which physical measurements triggered the alert" (`inference/explainability.py:13`) overstates it. And given section 3, band contrast on three of the six planes is a fixed linear function of the other three, so a "SWIR dominated" signature is arithmetic about RGB.

There is also a small live bug: `background_mask` is built to exclude the target box from the background region (`inference/explainability.py:145`, `:151`) and then never applied, because `bg_means` averages the whole region including the box (`inference/explainability.py:156`). The contrast is therefore target-versus-neighbourhood-including-target, which biases every contrast toward zero, more so for large boxes.

> ### The interview answer
>
> "The ground segment's job is to turn a brief into something an operator can act on, and its risk is that the reasoning layer is a language model. The evaluation harness is where I spent the effort, because the metric it replaced could not fail: it compared the number of detections to the number of assessments and returned a perfect score, which it happily did for briefs that substituted classes, placed objects four hundred kilometres away, fabricated every confidence value, and downgraded a critical scene to nominal. Now there are six independent axes: schema validity, entity grounding, coordinate fidelity, numeric fidelity, citation validity, and policy consistency. Entity matching is geodesic rather than positional, because the model is free to reorder its output and zipping by index manufactures fake substitutions. Policy deviation is scored asymmetrically, twice the penalty for under-escalating as for over-escalating. And the composite is the minimum across axes, not the mean, because a brief that invents coordinates is not redeemed by having valid JSON."

**Concepts you now own.** *Designing an evaluation that can fail*, which starts by writing down the specific wrong outputs you want to catch and checking your metric against each of them. *Composite scores hide dead components*, so use a minimum or report the parts. *Grounded generation fails silently in the retrieval layer*, so index integrity, query-versus-document asymmetry, and vector normalisation are correctness concerns and not tuning knobs. This is *DMLS* chapter 6's offline evaluation material extended to generative components, where the classic accuracy metrics do not apply and the failure modes are semantic.

---

## 9. Measurement as architecture

The old headline was **85,000:1**. Here is where it came from:

```
  100 MB Sentinel-2 scene
  ───────────────────────  =  85,000 : 1
  ~1.2 KB brief for ONE tile
```

The numerator is a whole scene. The denominator describes one 640 px tile. A Sentinel-2 10 m band is 10,980 px square, which tiles into `ceil(10980/640)² = 18² = 324` tiles. **The ratio was inflated by 324x by a coverage mismatch**, and nothing in the arithmetic was wrong: every number was correct, and they described different things.

The class that computes it now says so in its own docstring, keeps the field because the dashboard reads it, and renames it `proto_vs_raw_scene_unnormalised` beside a `..._normalised` figure that charges the scene the full 324 briefs needed to describe it (`inference/serialization_utils.py:264`). Its printed report labels the legacy line "do not quote".

That is the first lesson of this section: **a ratio is a claim about two things, and most bad ratios are correct arithmetic over mismatched denominators.**

### The same error again, in the denominator that replaced it

Fixing the coverage mismatch did not fix the denominator. It fixed *how much ground* the denominator covered and left *what the denominator was made of* untouched, and that second half was wrong for longer.

Every compression claim this project made divided by **9,830,400 B**: one tile as a float32 array. That number is correct and it is not a downlink cost. It is a working-set size, what the inference engine allocates to hold a tile while the model looks at it. Nothing transmits an uncompressed float32 buffer. A spacecraft compresses first, and the standard it compresses with is CCSDS 123.0-B-1.

So the alternative OSP was beating was not a codec. It was the absence of one.

```
   priced as              per tile        20-tile corpus     ratio vs 8,266 B
   ─────────────────────  ─────────────   ──────────────     ────────────────
   float32 in memory      9,830,400 B     196,608,000 B          23,785 : 1
   lossless PNG, uint16   2,546,639 B      50,932,783 B           6,162 : 1
   CCSDS 123.0-B-1          611,707 B      12,234,137 B           1,480 : 1
```

**16x, from the top row to the bottom.** The per-tile figure falls from 63,279:1 to 3,938:1 by the same factor.

What makes this one worth recording rather than merely fixing is that the repository already contained the correct reasoning and did not apply it. `ground/rate_distortion.py` had refused to price raw at float32 since it was written, on the explicit grounds that doing so "would inflate OSP's advantage by roughly 2x for free" — and then priced it as PNG, a codec no spacecraft flies, while the README divided by float32 anyway. Three prices for one tile, in one repository, differing by 16x end to end: the experiment was strict, the headline was not, and nothing forced them to agree.

`ground/ccsds123.py` now implements the standard, `orbital/downlink.py` defaults to `RAW_TILE_BYTES_CCSDS` and names `RAW_TILE_BYTES_FLOAT32` separately so the distinction cannot be made by accident, `data/briefs/manifest.json` records `raw_ccsds_bytes` per tile beside `wire_bytes` per brief, and `tests/test_raw_pricing.py` fails if the quoted ratio and the committed artifact drift apart.

One note in the other direction, because this baseline is generous to the opponent and that should be said before someone else says it: CCSDS 123 reaches **1.99 bits per sample** on this corpus, which is better than it would manage on a real instrument. Five of these six bands are linear functions of the first three (`prep_manifest.json` states this outright), so the inter-band predictor is exploiting redundancy a measuring sensor would never supply. On genuine multispectral data the raw side costs more and OSP's margin is wider than 1,480x. The number reported is the unkind one.

The second lesson is worse than the first, and it survives fixing both denominators.

### Why a ratio is the wrong question entirely

The honest per-tile ratio on the current corpus is 3,938:1 (155 B protobuf against a 0.61 MB CCSDS-compressed tile). The previous synthetic corpus reported 43,497:1. **The number went up, and that is not good news.** Nothing about the encoding improved. These 20 tiles average 1.05 detections per brief where the synthetic scenes were denser, and an emptier brief is a smaller brief. The largest per-scene ratio in the corpus belongs to a brief containing **zero detections**, because an empty brief is nearly free.

So a compression ratio partly measures how empty your scenes happened to be. That is a property of the dataset, not of the method.

And it answers the wrong question anyway. An operator does not ask how small a brief is. They ask: *given the bytes this pass affords, how much of what is down there will I know about?*

### The experiment

`ground/rate_distortion.py` fixes a byte budget and spends it three ways:

```
  raw       lossless six uint16 planes under PNG. Ground re-runs the detector.
  jpeg(q)   RGB at quality q. Ground re-derives the bands and detects.
  brief(c)  onboard detections above confidence c, as wire-format briefs.
```

then counts what the ground knows about the **whole corpus**:

```python
# ground/rate_distortion.py:290
cum_cost = np.cumsum(s.cost)
cum_tp   = np.cumsum(s.tp)
for b in budgets:
    n = int(np.searchsorted(cum_cost, b, side="right"))
    tp = int(cum_tp[n - 1]) if n else 0
    points.append({"recall": tp / total_gt, ...})
```

`total_gt` is every labelled object in the corpus, including objects on tiles that never fit the budget.

**That denominator is the entire experiment.** Score only the tiles that were delivered and every strategy trends to 1.0, which is exactly the flattering non-result this replaces: a codec that preserves a tile perfectly but only affords three tiles has still lost everything on the fourth. `tests/test_pipeline.py:925` pins the accounting with a hand-checkable case (three tiles, two objects each, 100 B apiece) and asserts that a budget below one tile recovers nothing, that recall is monotonic in budget, and that it saturates at 1.0 without exceeding it.

### The result

`MEASURED`, `model/artifacts/rate_distortion.json`. 1,000 held-out DOTA tiles sampled at even stride, 9,472 labelled objects, contact budget 32.0 kbps x 5.0 min = 1,200,000 B.

| Strategy | Bytes/tile | Recall | Precision | Tiles per contact |
| :--- | ---: | ---: | ---: | ---: |
| **raw, lossless (CCSDS 123)** | **592,934** | 0.862 | 0.920 | **2.0** |
| raw, lossless (PNG) | 2,366,944 | 0.862 | 0.920 | 0.5 |
| JPEG q75 | 53,317 | 0.828 | 0.920 | 22.5 |
| JPEG q30 | 26,079 | 0.814 | 0.925 | 46.0 |
| JPEG q2 | 8,291 | 0.507 | 0.846 | 144.7 |
| brief @ 0.20 | 951 | 0.893 | 0.873 | 1,262 |
| **brief @ 0.35** | **894** | **0.862** | **0.920** | **1,343** |
| brief @ 0.65 | 755 | 0.707 | 0.974 | 1,590 |
| brief @ 0.80 | 273 | **0.000** | **0.000** | 4,396 |

**The brief at confidence 0.35 ties raw lossless exactly.** 0.8622 recall and 0.9204 precision in both rows, to four decimal places, for 1/663rd of the bytes.

Two raw rows, because the codec is the whole difference. Both deliver identical pixels and therefore identical detections; PNG charges four times as much for them. PNG filters bytes within a plane and cannot see the band axis at all, while CCSDS 123 predicts each band from the bands beside it, which on a six-band cube derived from three is most of the available redundancy. The PNG row stays in the table because every earlier version of this document quoted it, and because a reader should be able to see what choosing the wrong lossless codec costs.

Not approximately, and it is not luck. The brief *is* the raw tile's detection result at the deployed threshold, and the ground station runs the same detector either way. `build_strategies` computes onboard detections once and the raw strategy filters that same result (`ground/rate_distortion.py:250`), because a lossless downlink means the ground sees identical pixels and therefore identical detections. The row is a tautology made visible, and it is the sharpest form of the architecture's argument: **at the deployed operating point, downlinking pixels buys the ground literally nothing over downlinking the answer.**

JPEG never reaches that recall at any quality: it peaks at 0.828 at q75, for 53.3 MB.

Three findings matter more than the headline gap.

**The detector has a confidence ceiling, and past it the brief goes silent.** At 0.80 the sweep emits zero detections across all 9,472 objects: not a degraded brief, an empty one, still costing 273 B/tile of envelope. Any operator threshold above roughly 0.7 silently downlinks nothing. That is a hard operating-envelope constraint and the single most important number in the table. The README states the model's maximum confidence "anywhere" is 0.683; a 96-tile stride sample finds **0.7143** (`MEASURED`, on `P0168_01440_00480.jpg`, an airplane), so 0.683 is a maximum over the tiles one sweep touched rather than a bound over the model. The operational conclusion is unchanged, and the phrasing is too strong.

**Briefs trade recall against precision, and neither is free.** Precision runs 0.754 at confidence 0.05 up to 0.974 at 0.65, while recall falls 0.921 to 0.707 across the same span. The knee is at 0.35, which is where the deployed threshold already sits. The old synthetic finding that briefs were lossless in precision was an artefact of a corpus where every object was a distinct geometric primitive.

> **Retracted.** An earlier version reported that heavy JPEG made the detector *hallucinate*: precision collapsing to 0.279 at q2 while recall fell to 0.562, so artefacts were manufacturing false objects. **That does not reproduce on real imagery.** At q2 on DOTA, recall falls to 0.507 and precision holds at 0.846, close to the raw baseline's 0.920 rather than collapsing. Heavy JPEG on real scenes hides objects, it does not invent them. The original result was a property of synthetic tiles, where compression artefacts on flat backgrounds resembled the drawn primitives the detector was trained on. Removed rather than restated.

### The sampling bug that returned zero storage tanks

`--limit` on this script and on `model/evaluate_detector.py` used to take the *first* N tiles. DOTA tile names sort by source image, so a prefix is a contiguous run of a handful of scenes. Measured on the real split:

```
all 3,677 tiles : ship 19,651   airplane 5,464   storage-tank 5,177   harbor 4,626
first 1,000     : ship 16,433   airplane   669   storage-tank     0   harbor 2,241
stride 1,000    : ship  5,270   airplane 1,528   storage-tank 1,414   harbor 1,260
```

`MEASURED`. The first 1,000 tiles hold **zero storage tanks**. An earlier draft of the rate-distortion table was built that way and was wrong, and the mAP it produced was a three-class mAP wearing a four-class label. Note the failure is silent again: nothing errors when a class has no ground truth, `average_precision` returns NaN for it (`model/evaluate_detector.py:104`) and the class is simply dropped from the mean (`model/evaluate_detector.py:279`).

Both tools now sample at even stride (`model/evaluate_detector.py:312`, `ground/rate_distortion.py:470`), which is deterministic, unlike a random sample, and reproduces the corpus class balance to within 0.7 points per class. The stride sample's 9,472 objects is exactly the `total_ground_truth_objects` in the committed artifact.

### The comparison is set up to be unkind to OSP

Three ways, each explicit in the code and repeated in the artifact's `caveats` list:

- **Raw is priced under CCSDS 123.0-B-1** (`ground/ccsds123.py`), the lossless standard for compressing image cubes on board a spacecraft, not as PNG and not as the 9.83 MB float32 array held in memory. CCSDS 123 costs about half what PNG does on this corpus, so this is the strongest fair opponent available rather than a convenient one. It is arguably *too* strong: it reaches 1.99 bits per sample here because five of the six bands are linear functions of the first three, redundancy a measuring instrument would not supply.
- **Briefs are priced as minified JSON** (`ground/rate_distortion.py:140`), when the protobuf they really ship in is 2.66x smaller.
- **Ground-side detection uses the same detector at the same threshold as onboard**, so the pixel strategies are never handicapped by a weaker analyst.

JPEG is the fair lossy baseline here for a specific reason: the six bands are a fixed linear map of RGB, so a tile's information content *is* its RGB, and compressing the RGB then re-deriving loses what the codec loses and nothing more. That would not hold for a sensor that measured its infrared independently, and the module says so at `ground/rate_distortion.py:41` rather than leaving it as an unstated advantage.

**What the curve cannot show**, stated alongside it (`ground/rate_distortion.py:63`): pixels can be re-analysed later, with a better model, for a question nobody has asked yet. A brief cannot. A brief is also unfalsifiable at the ground: if the onboard detector missed something, no amount of ground processing recovers it, and nothing in the brief reveals that it happened. The plot measures one axis of value and the architecture trades away another.

### The green suite that could not go red

The deepest measurement failure in this repository was not a number. It was the runner.

`tests/test_pipeline.py` used to be both a standalone script and a pytest module. It carried its own runner: a `@run_test` decorator that caught exceptions and recorded PASS/FAIL/SKIP, ANSI colour constants, a results accumulator, a hand-maintained list of every test function, and a summary block that set the exit code. That is right for a standalone runner, where you want the remaining tests to execute and the report to be complete. It is exactly wrong for pytest, which records a pass unless an exception propagates.

For most of this file's life only the first half was true. **Every test in it reported PASS under pytest no matter what it asserted, including a deliberately failing probe.** The first fix was a `_UNDER_PYTEST` flag that re-raised when pytest was driving — correct, and still a hand-rolled runner sitting between the assertions and the thing that reports them.

The runner is gone now. The tests are plain `def test_*` functions, skips are `pytest.skip`, and the report, the colours, the timing, the exit code and the discovery all come from pytest, which is what the other five suites in `tests/` already did. The manual registry is the part worth naming: a test function that was written but never added to that list would never have run, and nothing would have said so. Discovery you maintain by hand is a place for tests to go missing silently, which is the same failure as a runner that cannot go red, one level up.

The failure generalises past this repository. A test harness is itself untested code, and the specific thing it must do, propagate a failure, is the thing you never exercise while everything passes. The cheap defence is the one used here: write a test that is supposed to fail and confirm the harness reports it as a failure. The same instinct produced `survived > 0` in section 7 and the `all` control row in the band-dropout sweep (`resilience/degradation.py:133`), which blanks every input band to confirm the harness bites before anyone reads the null results above it.

### The reproduction check, and the honest state of it

```bash
python tools/verify_docker_repro.py --check-only   # prerequisites, no build
python tools/verify_docker_repro.py                # build, regenerate, diff
```

`data/briefs/` is the corpus the dashboard serves, and the README says it is reproducible rather than hand-authored. That claim had never been executed end to end: the root Dockerfile had not been built since Debian dropped `libgl1-mesa-glx`, so `docker build` failed outright. A reproducibility claim nobody has run is a claim, not a property.

The result is not a clean pass, and the diagnosis is the interesting part.

On the host, regeneration reproduces the committed corpus **20/20**. In the container it reproduces **7/20** exactly. The other 13 carry the same tiles and the same detection counts, with confidences differing by at most one step on the INT8 score ladder, about 0.037. Bisected:

```
  the ONNX graph      bit-identical in both   (same output hash on fixed input)
  the JPEG decode     bit-identical in both
  the derived 6-band tile                     DIFFERS  ~1e-8 relative
```

`rgb_to_6band` builds B11 and B12 with `cv2.resize`, and `INTER_AREA` agrees across environments while `INTER_LINEAR` does not, because OpenCV dispatches a different SIMD kernel against the container's CPU feature set. Against an FP32 detector a 1e-8 relative difference would not be observable. Against an INT8 one it is: quantisation snaps that hair's width onto the next rung of a discrete score ladder, and a detection sitting on the 0.35 threshold can cross it.

That is a property of quantised inference worth knowing rather than a defect in the container. The check enforces structural agreement, reports the numeric drift with its cause, and `--strict` demands bit-equality and currently fails, which is the honest state of it.

Note the scope carefully. `meta.inference_ms` is excluded because a wall-clock number that matched would mean the run did not really happen. And the model artifact and the tile split are bind-mounted read-only rather than baked in, so what is verified is *given the artifact and the split, the container regenerates `data/briefs/`*, not *a clean clone regenerates it from nothing*.

> ### The interview answer
>
> "The compression claim used to be a single ratio, eighty-five thousand to one, and it was correct arithmetic over mismatched denominators: a whole hundred-megabyte scene divided by one tile's brief, when a scene is three hundred and twenty-four tiles. But fixing the coverage does not fix the metric, because a ratio still partly measures how empty your scenes were. The largest ratio in my corpus belongs to a brief with zero detections. So I replaced it with a rate-distortion curve: fix a byte budget, spend it three ways, raw lossless, JPEG across a quality sweep, and briefs across a confidence sweep, and count what the ground knows about the *whole* corpus, with objects on tiles that never fit counted as missed. That denominator is the entire experiment: score only what was delivered and every strategy trends to one. The result is that the brief at confidence 0.35 ties raw lossless exactly, 0.862 recall and 0.920 precision, for one two-thousand-six-hundredth of the bytes, because the brief *is* the raw tile's detection result at the deployed threshold. And it exposed a hard constraint nobody chose: above about 0.7 confidence this detector emits nothing at all, so an operator threshold of 0.8 silently downlinks empty briefs forever."

**Concepts you now own.** *Operating-point curves instead of single-metric claims*, and specifically choosing the denominator that matches the decision being made. *Baselines must be constructed to be unkind to you*, priced generously and analysed with the same model. *Sampling design is a correctness concern*: a prefix is not a sample, and a class that vanishes from your evaluation set does not raise an error. And *your test harness is untested code until you make it go red on purpose*. These are *DMLS* chapter 6 (evaluation, baselines, slice-based evaluation) and chapter 2 (choosing the objective that matches the decision), and the sampling lesson recurs everywhere data is ordered by anything correlated with the label.

---

## What I would build next, and why I did not

Engineering judgment, not a wishlist. For each: what it costs, and what question it would actually settle.

**Real 10 m multispectral data, with labels.**
*Cost:* weeks. Sourcing Sentinel-2 L2A scenes with vessel or infrastructure labels at 10 m is the hard part, not the training. It would also lower the headline accuracy numbers, because objects arrive one to two orders of magnitude smaller than DOTA delivers them.
*Settles:* the only claim in this project I actually want and cannot make. Not "does the detector work" but "does the SWIR argument hold on measured spectra". Every band-dropout result here is a measurement about a linear map, and no amount of care makes it a measurement about infrared physics. It would also collapse the GSD mismatch in section 4, so briefs could carry `geolocation: real` honestly.
*Why not yet:* it would not change what this round was about, which is closing the loop between real orbital mechanics and a real resource decision. Doing it half-heartedly, on a handful of hand-labelled scenes, would produce a number too noisy to compare against 3,677 tiles and would read as a stronger claim than it was.

**An integrity check on the wire.**
*Cost:* an afternoon. A CRC-32 or a truncated HMAC as a proto field, computed over the serialised message, is 4 to 8 bytes on a 155 byte brief: 3 to 5% of the payload.
*Settles:* the measured gap in section 7, where a single flipped byte survives ingest roughly half the time as a well-formed wrong observation. It converts an undetectable corruption into a detectable one, which is the difference between a false observation and a lost one, and the ingest layer already knows what to do with a lost one.
*Why not yet:* honestly, because measuring the gap was more interesting than closing it, and because the fix is uninteresting enough that it should come with the two other wire-format changes it belongs beside: `sint32` for bbox coordinates, and a `degraded` field so the flag in section 2 survives the encoding. Those three together are one commit, and I would rather ship them as a versioned schema change than as three patches.

**Quantization-aware training.**
*Cost:* a retrain, so roughly the 3 h 11 m the DOTA run took on a P100, plus the work to insert fake-quant nodes into a 6-channel stem that Ultralytics does not expect.
*Settles:* whether the 0.9 point mAP drop at IoU 0.5 and 3.2 points at strict IoU is inherent to INT8 for this architecture, or an artifact of post-training calibration. QAT typically recovers most of the strict-IoU loss, and strict IoU is where this model loses the most, which suggests box regression is the part quantization hurts.
*Why not yet:* the PTQ number is not the binding constraint. At 51.4 ms against a 400 ms budget there is 7.8x of margin, and the accuracy that matters operationally is recall at the deployed threshold, where the brief already ties raw lossless. QAT would improve a number nobody is being limited by. It moves up the list the moment a smaller model or a tighter budget makes INT8 accuracy the constraint.

**A real link budget.**
*Cost:* not the arithmetic, which is an afternoon. The inputs. Antenna gains, noise figures, modulation and coding, pointing losses, and a real ground station's G/T.
*Settles:* how optimistic `PassEfficiency` is. Today the derating is four constants from peak elevation, 0.45 grazing to 0.90 excellent (`orbital/downlink.py:106`), against a real geometry where slant range runs about 2,000 km at the 10 degree mask versus about 700 km overhead, roughly 9 dB of extra path loss. A real budget would make the byte allowance a function of elevation *through the pass* rather than a scalar applied to the whole window, which would matter most for exactly the grazing passes the scheduler currently treats most generously.
*Why not yet:* a proper budget needs numbers this project has no basis to invent, and inventing them would dress an assumption as engineering. The current model is wrong in a way that is visible, named, adjustable and recorded in the audit trail, which is the honest version of not knowing.

**ARM hardware measurement.**
*Cost:* a Pi 5 or an Orin Nano, and a day.
*Settles:* whether the latency claim transfers. The current numbers are x86 held to two cores and 4 GB via cgroups: p99 end to end 307.65 ms against the platform's 400 ms budget (`MEASURED`, `docs/latency/constrained_2cpu_4gb.json`). That is a resource-constrained proxy and nothing more. ARM has a different instruction mix, different SIMD width, and ONNX Runtime dispatches different kernels, so these numbers will not extrapolate.
*Why not yet:* it is the cheapest item on this list and the one I would do first. It settles a claim I currently have to caveat in every direction, and unlike the others it requires no new data, no retraining, and no assumptions.

---

## The hardest questions a reviewer could ask

**1. "Aerial is not orbital. Your GSD is off by one to two orders of magnitude. Why should I believe any of this transfers?"**

You should not, and the repository refuses to let you. DOTA is roughly 0.1 to 1 m ground sample distance; the orbital layer models 10 m. Objects arrive between 10 and 100 times larger than a Sentinel-2 pass would deliver them, which is precisely the regime where small-object detection is easy. `tools/generate_briefs.py:103` refuses to mint Sentinel-2 footprint fields from these tiles without an explicit flag, and with the flag every brief's provenance says the geolocation is approximate and why. What transfers is the engineering: the byte budget, the quantization, the scheduler, the authority boundary, the fault behaviour. What does not transfer is the accuracy number. 0.880 mAP is a real measurement on real photographs of the right four classes, and it is not a claim about orbital imagery.

**2. "Every one of your six bands is a linear function of the other three. Isn't the whole multispectral story dead?"**

The architectural motivation is dead as a *demonstrated* advantage, yes. B8, B11 and B12 are fixed linear maps of R, G and B (`data/synthetic_bands.py:105`), the singular values show four significant components and then a cliff, and a convolution can form any of those mixes itself. So the infrared planes carry no information the visible bands did not already contain, and no result here validates the SWIR-through-haze argument.

Two things complicate the obituary. Dropping B11 and B12 together does cost 0.046 mAP on real imagery, because a network trained for 32 epochs learned to lean on channels it did not strictly need, and removing them at inference time is a distribution shift rather than an information loss. And the two costliest bands to drop are B2 and B4, plain visible channels, which is exactly what the arithmetic predicts and not the shape the earlier synthetic-only measurement suggested. Meanwhile the derived bands cost real latency: at two cores, `rgb_to_6band`'s two resize calls make preprocessing more expensive than inference at p50.

The honest summary: real channel surgery, real INT8 calibration across six planes, real band-dropout resilience, built correctly for a sensor this project does not have.

**3. "Your INT8 calibration tiles come from the split you score INT8 on. Isn't that leakage?"**

Yes, and the size is stated: 32 tiles out of 3,677, so roughly 0.9% of the scoring set was seen by the calibrator. Calibration fits activation ranges, not weights, so the effect is second-order: it can only make the observed activation ranges slightly better matched to those 32 tiles, and activation range is a much coarser quantity than a decision boundary. But it is not zero, and the INT8 column is not independent of its calibration data. The fix is trivial (calibrate on training tiles, which `train.py:131` already suggests as the default `--calib` path) and has not been redone because the artifact would have to be regenerated and every downstream number with it.

**4. "Your container reproduces 7 of 20 briefs bit-exactly. Isn't your reproducibility claim broken?"**

The strict version of it is, and `--strict` fails to say so rather than being quietly dropped. What holds is: same tiles, same detection counts, confidences differing by at most one step on the INT8 score ladder, about 0.037. The cause is traced rather than assumed: the ONNX graph is bit-identical in both environments and so is the JPEG decode, but `cv2.resize` with `INTER_LINEAR` dispatches a different SIMD kernel against the container's CPU feature set, and the resulting 1e-8 relative difference in the derived bands crosses an INT8 quantisation step. Against an FP32 detector it would be invisible. So this is a property of quantised inference, and the correct response is either to accept structural reproducibility and report the drift, which is what the tool does, or to pin the SIMD path. I chose the first because the second hides the lesson.

**5. "`_decide_ovv` accepts a language model's output into an action list. So the LLM is in the control loop after all."**

It is the one place a model output reaches a field named "action", and section 6 states the exact boundary. What it is not: it never reaches `DownlinkScheduler.plan`, which has no parameter for it; it never changes an alert level, because reconciliation cannot de-escalate; and in this repository an `OVVRequest` becomes a mission-log line and a dashboard card, with nothing that uplinks it. What it is: a model-authored proposal tagged `source="llm"` in the same list as rule-authored ones, and because the list is capped at three sorted by a priority number the model supplies itself, it can outbid a policy request for the last slot. That is a real channel, it is narrow, and the fix is to sort by `(source != "policy", priority)`. Everything else about the boundary is structural and tested. This one is a convention, and I would not ship it on a vehicle that actually acted on OVV requests.

**6. "Your `skyroot-oam` profile is invented. Aren't you evaluating against a straw man you sized to make your work look necessary?"**

The profile is `DERIVED` and every field says so, including in the dashboard header. It is a representative envelope for a launch-vehicle upper-stage compute class, sized an order of magnitude below `moi-1a` deliberately so the INT8 and compression work has to genuinely matter. That is a fair objection to how the profile was chosen.

The defence is that the *binding* number is no longer the invented one. The link budget's `contact_minutes_per_orbit = 5.0` was a guess, and the propagated geometry says 10.155 minutes, so the guess was pessimistic by 2x and the computation replaced it in the plan. The scheduler's behaviour on the current corpus is not a straw man either: 20 of 20 briefs fit with 99.6% of the window unused. If anything, the current corpus makes the scheduler look *unnecessary*, which is the opposite of the accusation and is why the rate-distortion curve rather than the scheduler carries the compression argument.

**7. "You claim determinism and your own reproduction check finds 13 mismatches. Which is it?"**

Both, at different scopes, and the scopes are worth separating. `bitwise_deterministic: true` in `quant_benchmark.json` means the same input through the same session in the same process produces byte-identical output, which is the property the assurance story needs: an on-orbit result is reproducible on the ground *given the same input tensor*. The container mismatch is upstream of the graph: the input tensors themselves differ by 1e-8 because band derivation dispatches different SIMD kernels. So the model is deterministic and the preprocessing is not, across CPU feature sets. That is the accurate statement, and it argues for pinning preprocessing rather than for doubting the model.

**8. "Your README quotes numbers your committed corpus does not reproduce. How much of the rest should I trust?"**

Three cases, all named in this document. The "15.8 KB, 104 detections, five oversize deferrals" figures are true of the previous corpus (`git show 55dd973:data/briefs/manifest.json`) and not of the current one, which schedules 20 of 20 with zero deferrals and totals 8,266 B. The committed `resilience/artifacts/degradation.json` *was* the synthetic-split sweep while the README's fault-tolerance table quoted an uncommitted DOTA re-run, so for a while the dashboard rendered a shape the README had already corrected. That one is closed: `degradation_dota.json` is committed, both documents read from it, and the dashboard prefers it. And "the maximum confidence this model produces anywhere is 0.683" is a maximum over one sweep's tiles; a 96-tile stride sample finds 0.7143.

What that says about the rest: every number in the README has a script, and the ones that drifted are the ones a script produces into a *file* while the prose was written by hand. The numbers backed by committed artifacts (`quant_benchmark.json`, `accuracy_int8.json`, `rate_distortion.json`, `docs/latency/*.json`) reproduce exactly, and I checked them for this document. The general defence is the one section 1 draws: derived prose needs a dependency graph, and the only ones available are code generation or an assertion in a test.

**9. "Nothing in your runtime path uses protobuf. Isn't the 155-byte number theatre?"**

It is a claim about the format, not about the files on disk, and the distinction matters in one direction only. `serialize_to_binary` appears nowhere outside `inference/serialization_utils.py` except in `tests/test_pipeline.py`: the engine writes JSON, the dashboard reads JSON, and the scheduler prices JSON. So today the corpus really is 2.66x larger than the headline.

Two things keep this from being decorative. The rate-distortion experiment, which carries the architectural argument, prices briefs as **minified JSON** and says so in its caveats, so the operating-point result is charged the larger number and the protobuf figure never enters it. And the schema is exercised end to end by the round-trip tests, which is how the missing `degraded` field in section 2 is discoverable at all. What is fair to say: the wire format is designed, tested and measured, and it has not yet been made the transport. Making it the transport is the same one commit as the integrity check and the `sint32` fix.

**10. "Your test suite reported PASS on everything for months. Why should I believe it now?"**

You should believe the tests exactly as far as the evidence goes, which is narrower than a green badge. A hand-rolled `@run_test` decorator swallowed exceptions, which is correct for a standalone runner and silently wrong under pytest, so every test in `tests/test_pipeline.py` reported PASS regardless of what it asserted, including a deliberately failing probe. That runner has since been deleted outright in favour of plain pytest functions. The current result is therefore a newer claim than the assertions are.

Three things go beyond "it is green". `tests/test_resilience.py` was checked by mutation rather than by running it: disabling the watchdog comparison fails two tests, and letting a degraded brief become the last-known-good fails a third. The band-dropout sweep carries an `all` control row that must score 0.000, so the null results above it are known to come from a harness that bites. And `test_a_single_flipped_byte_can_survive_ingest_undetected` asserts in both directions, so it fails if ingest validates nothing *and* if the documented gap ever silently closes. Those are the parts of the suite I would stand behind without the badge.

---

## Appendix: reproducing the full pipeline

Training from scratch, on the synthetic corpus, exercises every stage the DOTA run does:

```bash
python train.py --export          # ~100 min on CPU (26 epochs, synthetic corpus)
```

The shipping weights are **not** from that path. They come from `tools/kaggle_train_dota.ipynb`, which prepares DOTA-v1.0 via `data/dota_prep.py` (11,046 train / 3,677 val tiles at 640 px, stride 480) and runs the same two-phase schedule for 32 epochs on a Tesla P100, 3 h 11 m wall clock. Reproducing locally is the download-and-export path:

```bash
# after downloading osp_dota_artifacts.zip from the notebook's Output tab
unzip osp_dota_artifacts.zip -d .
python satellite_export.py --weights model/artifacts/osp_best.pt --calib val/images
python model/evaluate_detector.py --onnx model/artifacts/osp_yolov8n_int8.onnx \
    --images val/images --labels val/labels
```

> The notebook pins `torch==2.5.1+cu121` when it detects a compute capability below sm_70. Kaggle's preinstalled cu128 build has no kernels for the P100's sm_60, and `torch.cuda.is_available()` returns `True` anyway, so the notebook launches a real kernel to check rather than trusting that flag.

> Training defaults to CPU deliberately. On Apple's MPS backend, the identical run that reaches `cls_loss 0.29 / mAP50 0.99` on CPU diverges to `cls_loss 6.0 / mAP50 0.02` with the same seed and the same data. That's a numerical bug, not a speed trade-off.

The orbital layer and the dashboard's Docker image:

```bash
python orbital/tle.py                  # element sets in the committed snapshot, with ages
python tools/refresh_tle.py            # fetch a new dated snapshot from CelesTrak

docker build -f deploy/Dockerfile -t osp-dashboard . && docker run --rm -p 8501:8501 osp-dashboard
```

That image installs `deploy/requirements-dashboard.txt`, not the root manifest, and serves the committed corpus rather than running the detector, which takes it from roughly 10 GB to 1.46 GB.

---

*Every number in this document names the script that regenerates it. If one does not reproduce, the document is wrong and the code is right.*
