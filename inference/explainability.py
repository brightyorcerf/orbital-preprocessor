"""
inference/explainability.py
────────────────────────────
Spectral explainability and uncertainty estimation for OSP detections.

Provides interpretable, band-level explanations for each detection:
  - Which spectral bands contributed most to the detection
  - Confidence decomposition (model score vs spectral strength)
  - Uncertainty factors that reduce reliability
  - Human-readable spectral fingerprint descriptions

This module makes the system academically defensible:
"The detection is explainable — we know WHICH physical measurements
triggered the alert and WHY the confidence is what it is."

No additional model inference required — all computations are deterministic
post-hoc analysis of the detection geometry and band statistics.

Usage:
    from inference.explainability import BandExplainer, UncertaintyEstimator
    explainer = BandExplainer()
    explanation = explainer.explain(anomaly_dict, tile_6ch, footprint)
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

# ── Band metadata ──────────────────────────────────────────────────────────────

BAND_META = {
    0: {"name": "B2 (Blue)",  "wavelength_nm": 490,  "role": "visible_blue",
        "material_sensitivity": "water_clarity, shallow_bathymetry"},
    1: {"name": "B3 (Green)", "wavelength_nm": 560,  "role": "visible_green",
        "material_sensitivity": "vegetation, sediment, cloud"},
    2: {"name": "B4 (Red)",   "wavelength_nm": 665,  "role": "visible_red",
        "material_sensitivity": "painted surfaces, oxide minerals"},
    3: {"name": "B8 (NIR)",   "wavelength_nm": 842,  "role": "near_infrared",
        "material_sensitivity": "vegetation_edge, wake_detection, water_boundary"},
    4: {"name": "B11 (SWIR1)", "wavelength_nm": 1610, "role": "swir1",
        "material_sensitivity": "metallic_structures, ship_hull, dry_soil"},
    5: {"name": "B12 (SWIR2)", "wavelength_nm": 2190, "role": "swir2",
        "material_sensitivity": "steel_alloys, high_temp_surfaces, lithology"},
}


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class BandContribution:
    """Contribution of a single spectral band to a detection."""
    band_index: int
    band_name:  str
    mean_value: float          # mean pixel value in bbox region [0, 1]
    contrast:   float          # bbox value vs surrounding background
    contribution_score: float  # normalized 0–1
    interpretation: str        # human-readable


@dataclass
class SpectralExplanation:
    """Full spectral explanation for one detected anomaly."""
    anomaly_type:       str
    lat:                float
    lon:                float
    conf:               float
    band_contributions: list[BandContribution]
    dominant_bands:     list[str]      # names of top-2 contributing bands
    spectral_signature: str            # English description of the signature
    confidence_breakdown: dict         # model_conf, spectral_strength, combined
    uncertainty_factors:  list[str]
    detection_quality:    str          # "high" | "medium" | "low"


@dataclass
class UncertaintyReport:
    """Uncertainty analysis for a full scene."""
    scene_id:       str
    cloud_factor:   float    # 0=no degradation, 1=fully degraded
    band_quality:   dict     # per-band availability score
    overall_quality: float   # 0–1
    factors:        list[str]
    recommendations: list[str]


# ── Band Explainer ─────────────────────────────────────────────────────────────

class BandExplainer:
    """
    Post-hoc spectral band contribution analysis for OSP detections.

    Computes how much each of the 6 bands "contributed" to a detection
    by analysing the spectral signature within the detection bounding box
    compared to the local background.
    """

    def explain(
        self,
        anomaly: dict,
        tile_6ch: np.ndarray,
        tile_size: int = 640,
    ) -> Optional[SpectralExplanation]:
        """
        Generate a spectral explanation for one detected anomaly.

        Args:
            anomaly:   Anomaly dict from OSPPayload.anomalies
            tile_6ch:  (H, W, 6) float32 tile in [0, 1]
            tile_size: tile pixel size (default 640)

        Returns:
            SpectralExplanation or None if bbox is invalid
        """
        bbox = anomaly.get("bbox_px", [])
        if len(bbox) != 4:
            return None

        x1, y1, x2, y2 = bbox
        ll = anomaly.get("lat_lon", [0.0, 0.0])
        conf = anomaly.get("conf", 0.0)
        atype = anomaly.get("type", "unknown")

        # Clamp to tile dimensions
        H, W = tile_6ch.shape[:2]
        x1 = max(0, min(x1, W - 1))
        x2 = max(x1 + 1, min(x2, W))
        y1 = max(0, min(y1, H - 1))
        y2 = max(y1 + 1, min(y2, H))

        # Extract target region and background region
        target_region = tile_6ch[y1:y2, x1:x2, :]  # (h, w, 6)
        if target_region.size == 0:
            return None

        # Background: expanded region around the bbox (exclude the bbox itself)
        margin = max(30, (x2 - x1) * 2)
        bx1 = max(0, x1 - margin)
        bx2 = min(W, x2 + margin)
        by1 = max(0, y1 - margin)
        by2 = min(H, y2 + margin)
        background_mask = np.ones((by2 - by1, bx2 - bx1), dtype=bool)
        # Mask out the target bbox from background
        rel_y1 = y1 - by1
        rel_y2 = y2 - by1
        rel_x1 = x1 - bx1
        rel_x2 = x2 - bx1
        background_mask[rel_y1:rel_y2, rel_x1:rel_x2] = False
        background_region = tile_6ch[by1:by2, bx1:bx2, :]

        # Compute per-band means
        target_means = target_region.reshape(-1, 6).mean(axis=0)
        bg_means     = background_region.reshape(-1, 6).mean(axis=0)
        contrast     = target_means - bg_means   # positive = brighter than background

        # Contribution score: normalized absolute contrast
        abs_contrast = np.abs(contrast)
        total_contrast = abs_contrast.sum() + 1e-8
        contrib_scores = abs_contrast / total_contrast

        # Build band contributions
        band_contribs: list[BandContribution] = []
        for i in range(6):
            meta = BAND_META[i]
            interp = self._interpret_band(i, contrast[i], target_means[i], atype)
            band_contribs.append(BandContribution(
                band_index          = i,
                band_name           = meta["name"],
                mean_value          = float(round(target_means[i], 3)),
                contrast            = float(round(contrast[i], 3)),
                contribution_score  = float(round(contrib_scores[i], 3)),
                interpretation      = interp,
            ))

        # Sort by contribution
        band_contribs_sorted = sorted(
            band_contribs, key=lambda b: b.contribution_score, reverse=True
        )
        dominant_bands = [b.band_name for b in band_contribs_sorted[:2]]

        # Spectral signature summary
        spectral_sig = self._describe_signature(band_contribs, atype, contrast)

        # Confidence breakdown
        spectral_strength = float(np.clip(
            abs_contrast[3:].mean() * 3.0, 0.0, 1.0  # SWIR+NIR strength
        ))
        confidence_breakdown = {
            "model_score":       round(conf, 3),
            "spectral_strength": round(spectral_strength, 3),
            "combined_estimate": round((conf + spectral_strength) / 2, 3),
        }

        # Uncertainty factors
        uncertainty = self._compute_uncertainty(tile_6ch, conf, contrast)

        # Detection quality
        quality = (
            "high"   if conf >= 0.75 and spectral_strength >= 0.3 else
            "medium" if conf >= 0.55 or spectral_strength >= 0.2 else
            "low"
        )

        return SpectralExplanation(
            anomaly_type        = atype,
            lat                 = ll[0],
            lon                 = ll[1],
            conf                = conf,
            band_contributions  = band_contribs,
            dominant_bands      = dominant_bands,
            spectral_signature  = spectral_sig,
            confidence_breakdown = confidence_breakdown,
            uncertainty_factors  = uncertainty,
            detection_quality    = quality,
        )

    def _interpret_band(
        self,
        band_idx: int,
        contrast: float,
        mean_val: float,
        anomaly_type: str,
    ) -> str:
        """Generate a physics-based interpretation for one band's signal."""
        sign = "elevated" if contrast > 0.02 else ("suppressed" if contrast < -0.02 else "neutral")

        interpretations = {
            0: f"B2 (Blue): {sign} — {'possible foam/wake scatter' if sign == 'elevated' else 'ocean absorption typical'}",
            1: f"B3 (Green): {sign} — {'bright object or cloud' if sign == 'elevated' else 'ocean typical'}",
            2: f"B4 (Red): {sign} — {'painted metal surface or dust' if sign == 'elevated' else 'typical'}",
            3: f"B8 (NIR): {sign} — {'wake or boundary signature' if sign == 'elevated' else 'absorbing (water/ocean typical)'}",
            4: f"B11 (SWIR1): {sign} — {'metallic hull or man-made material' if sign == 'elevated' else 'water (expected low SWIR)'}",
            5: f"B12 (SWIR2): {sign} — {'steel/alloy surface' if sign == 'elevated' else 'ocean (expected low SWIR)'}",
        }
        return interpretations.get(band_idx, f"Band {band_idx}: {sign}")

    def _describe_signature(
        self,
        bands: list[BandContribution],
        anomaly_type: str,
        contrast: np.ndarray,
    ) -> str:
        """Build a one-line English spectral fingerprint description."""
        swir_elevated = (contrast[4] > 0.05 or contrast[5] > 0.05)
        nir_elevated  = contrast[3] > 0.03
        vis_elevated  = any(contrast[i] > 0.05 for i in range(3))

        parts = []
        if swir_elevated:
            parts.append("strong SWIR signature (metallic/man-made material confirmed)")
        if nir_elevated:
            parts.append("NIR boundary contrast (consistent with moving vessel wake)")
        if vis_elevated and not swir_elevated:
            parts.append("visible-band dominant (painted surface or cloud contamination)")
        if not parts:
            parts.append("weak spectral contrast across all bands (low-confidence detection)")

        type_note = {
            "ship":         "Consistent with vessel hull reflectance profile.",
            "airplane":     "Consistent with aluminium fuselage SWIR signature.",
            "harbor":       "Consistent with concrete/steel port infrastructure.",
            "storage-tank": "Consistent with metallic tank surface reflectance.",
        }.get(anomaly_type, "")

        return "; ".join(parts) + (f" {type_note}" if type_note else "")

    def _compute_uncertainty(
        self,
        tile_6ch: np.ndarray,
        conf: float,
        contrast: np.ndarray,
    ) -> list[str]:
        """Identify factors that increase detection uncertainty."""
        factors = []

        # Cloud contamination proxy (B3 brightness)
        b3_mean = tile_6ch[:, :, 1].mean()
        if b3_mean > 0.55:
            factors.append(f"Cloud contamination (B3 mean={b3_mean:.2f} > 0.55 threshold)")

        # Low model confidence
        if conf < 0.55:
            factors.append(f"Low model confidence ({conf:.0%} < 55% threshold)")

        # Weak SWIR signal
        swir_contrast = (abs(contrast[4]) + abs(contrast[5])) / 2
        if swir_contrast < 0.03:
            factors.append("Weak SWIR contrast — no metallic signature confirmation")

        # High NIR variability (water roughness / glint)
        nir_std = tile_6ch[:, :, 3].std()
        if nir_std > 0.2:
            factors.append(f"High NIR variability (std={nir_std:.2f}) — possible sun glint")

        if not factors:
            factors.append("No significant uncertainty factors identified")

        return factors

    def explain_batch(
        self,
        anomalies: list[dict],
        tile_6ch: np.ndarray,
    ) -> list[SpectralExplanation]:
        """Explain all anomalies in a scene."""
        return [
            exp for a in anomalies
            if (exp := self.explain(a, tile_6ch)) is not None
        ]


# ── Uncertainty Estimator ──────────────────────────────────────────────────────

class UncertaintyEstimator:
    """
    Scene-level sensing uncertainty estimator.

    Quantifies how much the current sensing conditions degrade the
    reliability of the detection pipeline's outputs.
    """

    def estimate(self, payload: dict, tile_6ch: Optional[np.ndarray] = None) -> UncertaintyReport:
        """
        Estimate overall sensing uncertainty for a scene.

        Args:
            payload:   OSP payload dict
            tile_6ch:  Optional (H, W, 6) tile for band-level analysis

        Returns:
            UncertaintyReport with quality scores and recommendations
        """
        scene_id  = payload.get("scene_id", "UNKNOWN")
        cloud     = payload.get("cloud_cover", 0.0)
        anomalies = payload.get("anomalies", [])

        # Cloud degradation factor (0=clear, 1=opaque)
        cloud_factor = min(1.0, cloud * 1.5)   # cloud degrades faster at higher coverage

        # Band quality scores (approximated from cloud cover if no tile available)
        band_quality = {}
        if tile_6ch is not None:
            for i in range(6):
                band = tile_6ch[:, :, i]
                # High std across scene = variable (possibly cloudy)
                quality = float(np.clip(1.0 - band.std() * 2.0, 0.2, 1.0))
                band_quality[BAND_META[i]["name"]] = round(quality, 2)
        else:
            # Simplified proxy from cloud cover
            for i in range(6):
                vis_penalty = cloud * 0.8 if i < 3 else cloud * 0.3  # SWIR more resilient
                band_quality[BAND_META[i]["name"]] = round(max(0.2, 1.0 - vis_penalty), 2)

        overall = float(np.mean(list(band_quality.values())))

        factors = []
        recommendations = []

        if cloud > 0.6:
            factors.append(f"Severe cloud cover ({cloud:.0%}) — visible bands severely degraded")
            recommendations.append("Schedule OVV for next clear pass")

        elif cloud > 0.3:
            factors.append(f"Moderate cloud cover ({cloud:.0%}) — reduce confidence thresholds")
            recommendations.append("Lower confidence acceptance threshold by 0.10–0.15")

        if not anomalies and cloud > 0.3:
            factors.append("Zero detections under moderate cloud — may reflect sensing limitation")
            recommendations.append("Do not classify as area-clear; re-observe when cloud clears")

        low_conf_anomalies = [a for a in anomalies if a.get("conf", 1.0) < 0.55]
        if low_conf_anomalies:
            factors.append(
                f"{len(low_conf_anomalies)} low-confidence detection(s) — "
                "SWIR corroboration recommended"
            )
            recommendations.append("Cross-validate low-conf detections with SWIR band analysis")

        if not factors:
            factors.append("No significant uncertainty factors — nominal sensing conditions")
            recommendations.append("Standard confidence thresholds apply")

        return UncertaintyReport(
            scene_id        = scene_id,
            cloud_factor    = round(cloud_factor, 2),
            band_quality    = band_quality,
            overall_quality = round(overall, 2),
            factors         = factors,
            recommendations = recommendations,
        )


# ── Formatting helpers ─────────────────────────────────────────────────────────

def format_explanation_for_prompt(explanation: SpectralExplanation) -> str:
    """
    Convert a SpectralExplanation to a compact LLM-injectable string.
    Used to provide spectral context in the ORION analyst prompt.
    """
    return (
        f"Spectral analysis [{explanation.anomaly_type.upper()} @ "
        f"({explanation.lat:.4f}°, {explanation.lon:.4f}°)]: "
        f"{explanation.spectral_signature} "
        f"Quality: {explanation.detection_quality}. "
        f"Dominant bands: {', '.join(explanation.dominant_bands)}. "
        f"Uncertainty: {'; '.join(explanation.uncertainty_factors[:2])}."
    )


# ── CLI demo ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    # Synthetic tile: 640×640×6 with a bright rectangular anomaly
    tile = np.random.uniform(0.05, 0.25, (640, 640, 6)).astype(np.float32)
    # Insert a "ship" (bright in SWIR, moderate in visible)
    tile[200:240, 310:360, 4] = 0.45  # B11 SWIR1 elevated
    tile[200:240, 310:360, 5] = 0.40  # B12 SWIR2 elevated
    tile[200:240, 310:360, 1] = 0.30  # B3 slightly elevated

    test_anomaly = {
        "type":    "ship",
        "lat_lon": [8.412, 77.821],
        "conf":    0.87,
        "bbox_px": [310, 200, 360, 240],
    }

    explainer = BandExplainer()
    explanation = explainer.explain(test_anomaly, tile)

    if explanation:
        print(f"\nSpectral Explanation: {explanation.anomaly_type.upper()}")
        print(f"Signature: {explanation.spectral_signature}")
        print(f"Quality:   {explanation.detection_quality}")
        print(f"Dominant:  {explanation.dominant_bands}")
        print(f"\nBand contributions:")
        for b in sorted(explanation.band_contributions,
                        key=lambda x: x.contribution_score, reverse=True):
            print(f"  {b.band_name}: score={b.contribution_score:.3f} | {b.interpretation}")

        print(f"\nConfidence: {explanation.confidence_breakdown}")
        print(f"\nUncertainty factors:")
        for f in explanation.uncertainty_factors:
            print(f"  • {f}")

    # Uncertainty report
    sample_payload = {
        "scene_id": "OSP-TEST",
        "cloud_cover": 0.35,
        "anomalies": [{"type": "ship", "conf": 0.51, "lat_lon": [8.4, 77.8], "bbox_px": [310,200,360,240]}],
    }
    estimator = UncertaintyEstimator()
    report    = estimator.estimate(sample_payload, tile)
    print(f"\nUncertainty Report: {report.scene_id}")
    print(f"  Overall quality: {report.overall_quality:.0%}")
    for f in report.factors:
        print(f"  • {f}")
    for r in report.recommendations:
        print(f"  → {r}")
