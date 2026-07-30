"""MedFuse in-hospital mortality task for MIMIC-IV (EHR time series + chest X-ray).

Produces MedFuse-ready samples directly from a :class:`~pyhealth.datasets.MIMIC4Dataset`:

    {"ehr": <[n_bins, F] time-series>, "cxr": <image path>, "cxr_mask": [1.0],
     "mortality": <0/1>}

The ``ehr`` and ``cxr`` keys match :class:`~pyhealth.models.MedFuse`'s default feature
keys, so the resulting ``SampleDataset`` feeds MedFuse with no model changes. The EHR
signal is configurable between ICU **vitals** (``chartevents``) and chemistry **labs**
(``labevents``); the observation window is a fixed post-admission window (paper default:
first 48h, resampled every 2h).

Paper:
    Hayat, N., Geras, K. J., & Shamout, F. E. (2022). MedFuse: Multi-modal fusion with
    clinical time-series data and chest X-ray images. MLHC 2022.

v1 (this class) requires **both** modalities (paired cohort). The missing-CXR (partial)
cohort is a deferred subclass that overrides :meth:`_resolve_cxr`.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl

from .base_task import BaseTask
from .mortality_prediction import MultimodalMortalityPredictionMIMIC4


# --------------------------------------------------------------------------- #
# Feature specs
# --------------------------------------------------------------------------- #
# ICU vitals -> MIMIC-IV `chartevents` itemids (d_items, itemids ~220xxx).
# Feature order is fixed so the produced matrix has a constant column count F.
VITAL_CATEGORIES: Dict[str, List[str]] = {
    "heart_rate": ["220045"],
    "sbp": ["220050", "220179"],            # arterial + non-invasive systolic
    "dbp": ["220051", "220180"],            # arterial + non-invasive diastolic
    "mbp": ["220052", "220181", "225312"],  # arterial + non-invasive + cuff mean
    "resp_rate": ["220210", "224690"],
    "spo2": ["220277"],
    "temperature": ["223761", "223762"],    # 223761 = Fahrenheit (converted), 223762 = Celsius
    "glucose": ["220621", "225664", "226537"],
    "gcs_eye": ["220739"],
    "gcs_verbal": ["223900"],
    "gcs_motor": ["223901"],
    "weight": ["226512", "224639"],
    "height": ["226730"],
}

# Per-itemid value transforms. Kept module-level (NOT stored on the task instance) so the
# task's ``vars()`` stays JSON-serializable/deterministic for set_task cache keying.
_ITEMID_TRANSFORM = {
    "223761": lambda f: (f - 32.0) * 5.0 / 9.0,  # Fahrenheit -> Celsius
}


class MedFuseMortalityMIMIC4(BaseTask):
    """MedFuse in-hospital mortality prediction on MIMIC-IV (paired EHR + CXR).

    One sample per qualifying ICU stay. The EHR feature is a fixed-window time-series
    matrix (``chartevents`` vitals or ``labevents`` labs); the CXR feature is the path to
    the last chest X-ray taken during the stay (loaded to a tensor by the ``image``
    processor). Mortality is the admission's ``hospital_expire_flag``.

    Args:
        ehr_source: ``"vitals"`` (chartevents) or ``"labs"`` (labevents). Default ``"vitals"``.
        window_hours: Observation window length after ICU ``intime``. Default 48.
        fixed_window: If True, emit a dense ``[n_bins, F]`` matrix on a fixed grid
            (``n_bins = window_hours / bin_hours``) via the ``tensor`` processor. If False,
            emit ``(timestamps, values)`` for the ``timeseries`` processor (variable length).
            Default True.
        bin_hours: Bin width for the fixed grid / resample rate for the variable path. Default 2.
        min_los_hours: Minimum ICU length of stay (computed from intime/outtime). Default 48.
        min_age: Minimum patient anchor_age. Default 18.
        require_cxr: If True (v1), skip stays without a CXR. Default True.
        image_size: Square resize for the CXR image processor. Default 224.
        cxr_event_type: Event type of CXR metadata events. Default ``"metadata"``.
    """

    task_name: str = "medfuse_mortality_mimic4"

    def __init__(
        self,
        ehr_source: str = "vitals",
        window_hours: float = 48.0,
        fixed_window: bool = True,
        bin_hours: float = 2.0,
        min_los_hours: float = 48.0,
        min_age: int = 18,
        require_cxr: bool = True,
        image_size: int = 224,
        cxr_event_type: str = "metadata",
    ) -> None:
        if ehr_source == "vitals":
            self.table = "chartevents"
            categories = VITAL_CATEGORIES
        elif ehr_source == "labs":
            self.table = "labevents"
            categories = MultimodalMortalityPredictionMIMIC4.LAB_CATEGORIES
        else:
            raise ValueError(f"ehr_source must be 'vitals' or 'labs', got {ehr_source!r}.")

        self.ehr_source = ehr_source
        self.window_hours = float(window_hours)
        self.fixed_window = bool(fixed_window)
        self.bin_hours = float(bin_hours)
        self.min_los_hours = float(min_los_hours)
        self.min_age = int(min_age)
        self.require_cxr = bool(require_cxr)
        self.image_size = int(image_size)
        self.cxr_event_type = str(cxr_event_type)

        # Derived, JSON-safe lookups (feature order fixed -> constant F).
        self.feature_names: List[str] = list(categories.keys())
        self.n_features: int = len(self.feature_names)
        self.n_bins: int = int(round(self.window_hours / self.bin_hours))
        self.all_itemids: List[str] = [i for ids in categories.values() for i in ids]
        self.itemid_to_feat: Dict[str, int] = {
            itemid: idx
            for idx, name in enumerate(self.feature_names)
            for itemid in categories[name]
        }

        if self.fixed_window:
            ehr_schema: Any = "tensor"
        else:
            ehr_schema = ("timeseries", {"sampling_rate": timedelta(hours=self.bin_hours)})
        self.input_schema: Dict[str, Any] = {
            "ehr": ehr_schema,
            "cxr": ("image", {"image_size": self.image_size}),
            "cxr_mask": "tensor",
        }
        self.output_schema: Dict[str, str] = {"mortality": "binary"}
        super().__init__()

    # ------------------------------------------------------------------ #
    # EHR extraction
    # ------------------------------------------------------------------ #
    def _parse_events(
        self, patient: Any, t0: datetime, t1: datetime
    ) -> List[Tuple[datetime, int, float]]:
        """Returns [(timestamp, feature_idx, value)] for in-window, in-spec measurements."""
        df = patient.get_events(
            event_type=self.table, start=t0, end=t1, return_df=True
        )
        if df is None or df.height == 0:
            return []

        icol, vcol = f"{self.table}/itemid", f"{self.table}/valuenum"
        df = df.select(
            pl.col("timestamp"),
            pl.col(icol).cast(pl.Utf8).alias("itemid"),
            pl.col(vcol).cast(pl.Float64, strict=False).alias("valuenum"),
        ).filter(pl.col("itemid").is_in(self.all_itemids))
        if df.height == 0:
            return []

        out: List[Tuple[datetime, int, float]] = []
        for ts, itemid, val in zip(
            df["timestamp"].to_list(),
            df["itemid"].to_list(),
            df["valuenum"].to_list(),
        ):
            if val is None or (isinstance(val, float) and math.isnan(val)):
                continue
            feat_idx = self.itemid_to_feat.get(itemid)
            if feat_idx is None:
                continue
            transform = _ITEMID_TRANSFORM.get(itemid)
            out.append((ts, feat_idx, transform(val) if transform else float(val)))
        return out

    def _build_fixed(
        self, events: List[Tuple[datetime, int, float]], t0: datetime
    ) -> Optional[List[List[float]]]:
        """Bins events onto a fixed [n_bins, F] grid (mean per bin), then forward-fills."""
        sums = np.zeros((self.n_bins, self.n_features), dtype=np.float64)
        counts = np.zeros((self.n_bins, self.n_features), dtype=np.float64)
        bin_seconds = self.bin_hours * 3600.0
        for ts, feat_idx, value in events:
            b = int((ts - t0).total_seconds() // bin_seconds)
            if 0 <= b < self.n_bins:
                sums[b, feat_idx] += value
                counts[b, feat_idx] += 1
        if counts.sum() == 0:
            return None
        matrix = np.divide(
            sums, counts, out=np.full_like(sums, np.nan), where=counts > 0
        )
        matrix = self._forward_fill(matrix)
        return matrix.tolist()

    def _build_variable(
        self, events: List[Tuple[datetime, int, float]]
    ) -> Optional[Tuple[List[datetime], np.ndarray]]:
        """One row per distinct timestamp (mean per feature); timeseries processor resamples."""
        per_ts: Dict[datetime, Tuple[np.ndarray, np.ndarray]] = {}
        for ts, feat_idx, value in events:
            s, c = per_ts.setdefault(
                ts,
                (np.zeros(self.n_features), np.zeros(self.n_features)),
            )
            s[feat_idx] += value
            c[feat_idx] += 1
        if not per_ts:
            return None
        timestamps = sorted(per_ts)
        rows = [
            np.divide(
                s, c, out=np.full_like(s, np.nan), where=c > 0
            )
            for s, c in (per_ts[ts] for ts in timestamps)
        ]
        return timestamps, np.vstack(rows)

    @staticmethod
    def _forward_fill(matrix: np.ndarray) -> np.ndarray:
        """Forward-fills NaNs down the time axis; leading NaNs become 0.0."""
        n_steps, n_features = matrix.shape
        for j in range(n_features):
            last = 0.0
            for i in range(n_steps):
                if np.isnan(matrix[i, j]):
                    matrix[i, j] = last
                else:
                    last = matrix[i, j]
        return matrix

    def _build_ehr(self, patient: Any, t0: datetime, t1: datetime):
        events = self._parse_events(patient, t0, t1)
        if not events:
            return None
        if self.fixed_window:
            return self._build_fixed(events, t0)
        return self._build_variable(events)

    # ------------------------------------------------------------------ #
    # CXR extraction (overridden by the v2 partial-cohort subclass)
    # ------------------------------------------------------------------ #
    def _resolve_cxr(
        self, patient: Any, intime: datetime, outtime: datetime
    ) -> Optional[Tuple[str, float]]:
        """Returns (image_path, mask) for the last CXR in the stay, or None to skip (v1)."""
        best_path: Optional[str] = None
        best_ts: Optional[datetime] = None
        for event in patient.get_events(event_type=self.cxr_event_type):
            ts = getattr(event, "timestamp", None)
            if ts is None or ts < intime or ts > outtime:
                continue
            path = getattr(event, "image_path", None)
            if not path:
                continue
            if best_ts is None or ts > best_ts:
                best_ts, best_path = ts, path
        if best_path is None:
            return None
        return best_path, 1.0

    # ------------------------------------------------------------------ #
    # Label
    # ------------------------------------------------------------------ #
    @staticmethod
    def _mortality(adm_flag: Dict[Any, Any], hadm_id: Any) -> Optional[int]:
        flag = adm_flag.get(hadm_id)
        if flag is None:
            return None
        try:
            return int(flag)
        except (TypeError, ValueError):
            return 1 if str(flag).strip() == "1" else 0

    @staticmethod
    def _parse_dt(value: Any) -> Optional[datetime]:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        try:
            return datetime.fromisoformat(str(value))
        except ValueError:
            return None

    # ------------------------------------------------------------------ #
    # Task entry point
    # ------------------------------------------------------------------ #
    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        demographics = patient.get_events(event_type="patients")
        if not demographics:
            return []
        try:
            age = int(demographics[0].anchor_age)
        except (TypeError, ValueError, AttributeError):
            age = None
        if age is not None and age < self.min_age:
            return []

        admissions = patient.get_events(event_type="admissions")
        adm_flag = {a.hadm_id: getattr(a, "hospital_expire_flag", None) for a in admissions}

        samples: List[Dict[str, Any]] = []
        for stay in patient.get_events(event_type="icustays"):
            intime = self._parse_dt(getattr(stay, "timestamp", None))
            outtime = self._parse_dt(getattr(stay, "outtime", None))
            if intime is None or outtime is None:
                continue
            if (outtime - intime).total_seconds() / 3600.0 < self.min_los_hours:
                continue
            t0 = intime
            t1 = intime + timedelta(hours=self.window_hours)

            ehr = self._build_ehr(patient, t0, t1)
            if ehr is None:
                continue

            resolved = self._resolve_cxr(patient, intime, outtime)
            if resolved is None:
                continue
            image_path, mask = resolved

            label = self._mortality(adm_flag, getattr(stay, "hadm_id", None))
            if label is None:
                continue

            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "stay_id": getattr(stay, "stay_id", None),
                    "ehr": ehr,
                    "cxr": image_path,
                    "cxr_mask": [float(mask)],
                    "mortality": label,
                }
            )
        return samples
