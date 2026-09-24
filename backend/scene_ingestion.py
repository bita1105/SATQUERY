from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    import rasterio
    RASTERIO_AVAILABLE = True
except ImportError:
    rasterio = None
    RASTERIO_AVAILABLE = False


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
RASTER_EXTENSIONS = {".tif", ".tiff"}


@dataclass
class SceneMetadata:
    filename: str
    format: str
    modality: str
    modality_source: str
    modality_confidence: float
    sensor: str
    sensor_source: str
    width: int
    height: int
    band_count: int
    dtype: str
    crs: str | None
    bounds: dict[str, float] | None
    resolution: dict[str, float] | None
    acquisition_time: str | None
    nodata: float | None
    is_georeferenced: bool
    metadata_available: bool
    statistics: dict[str, Any]
    warnings: list[str]


@dataclass
class Scene:
    metadata: SceneMetadata
    data: Any
    profile: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self.metadata)
        result["profile"] = self.profile
        return result


def _norm(text: str | None) -> str:
    return str(text or "").strip().lower()


def infer_modality_and_sensor(
    filename: str,
    tags: dict[str, Any] | None = None,
):
    text = _norm(filename)

    if tags:
        text += " " + _norm(" ".join(f"{k}={v}" for k, v in tags.items()))

    if any(x in text for x in (
        "sentinel-1", "sentinel1", "s1a_", "s1b_", "s1c_",
        "sar", "grd", "slc",
    )):
        sensor = "Sentinel-1" if ("sentinel-1" in text or "sentinel1" in text) else "unknown"
        confidence = 0.92 if sensor == "Sentinel-1" else 0.72
        return "sar", sensor, confidence, "filename_or_metadata", "filename_or_metadata"

    if any(x in text for x in (
        "sentinel-2", "sentinel2", "s2a_", "s2b_", "s2c_",
        "optical", "l2a", "l1c",
    )):
        sensor = "Sentinel-2" if ("sentinel-2" in text or "sentinel2" in text) else "unknown"
        confidence = 0.92 if sensor == "Sentinel-2" else 0.72
        return "optical", sensor, confidence, "filename_or_metadata", "filename_or_metadata"

    return "unknown", "unknown", 0.0, "not_available", "not_available"


def _stats(data: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(data)
    finite = arr[np.isfinite(arr)]

    if finite.size == 0:
        return {"min": None, "max": None, "mean": None, "std": None}

    return {
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
    }


def _read_geotiff(path: Path) -> Scene:
    if not RASTERIO_AVAILABLE:
        raise RuntimeError(
            "Rasterio is required for GeoTIFF ingestion. "
            "Install it in the ASTRA backend environment."
        )

    with rasterio.open(path) as src:
        data = src.read()
        tags = src.tags()

        crs = src.crs.to_string() if src.crs else None
        bounds = None

        if src.bounds:
            bounds = {
                "left": float(src.bounds.left),
                "bottom": float(src.bounds.bottom),
                "right": float(src.bounds.right),
                "top": float(src.bounds.top),
            }

        resolution = None
        if src.res:
            resolution = {
                "x": abs(float(src.res[0])),
                "y": abs(float(src.res[1])),
            }

        modality, sensor, mconf, msource, ssource = infer_modality_and_sensor(
            path.name, tags
        )

        warnings = []

        if crs is None:
            warnings.append("CRS is missing.")
        if bounds is None:
            warnings.append("Spatial bounds are unavailable.")
        if modality == "unknown":
            warnings.append("Sensor/modality could not be identified from metadata or filename.")

        acquired = (
            tags.get("ACQUISITION_TIME")
            or tags.get("DATETIME")
            or tags.get("TIFFTAG_DATETIME")
        )

        nodata = None
        try:
            if src.nodata is not None and np.isfinite(src.nodata):
                nodata = float(src.nodata)
        except (TypeError, ValueError):
            pass

        profile = {
            "driver": src.driver,
            "count": int(src.count),
            "width": int(src.width),
            "height": int(src.height),
            "dtype": str(data.dtype),
            "crs": crs,
            "transform": [float(v) for v in src.transform[:6]],
            "bounds": bounds,
        }

        metadata = SceneMetadata(
            filename=path.name,
            format=src.driver,
            modality=modality,
            modality_source=msource,
            modality_confidence=round(mconf, 3),
            sensor=sensor,
            sensor_source=ssource,
            width=int(src.width),
            height=int(src.height),
            band_count=int(src.count),
            dtype=str(data.dtype),
            crs=crs,
            bounds=bounds,
            resolution=resolution,
            acquisition_time=acquired,
            nodata=nodata,
            is_georeferenced=(crs is not None and bounds is not None),
            metadata_available=bool(tags),
            statistics=_stats(data),
            warnings=warnings,
        )

    return Scene(metadata=metadata, data=data, profile=profile)


def _read_standard_image(path: Path) -> Scene:
    with Image.open(path) as img:
        rgb = img.convert("RGB")
        hwc = np.asarray(rgb)
        data = np.transpose(hwc, (2, 0, 1)).copy()

        modality, sensor, mconf, msource, ssource = infer_modality_and_sensor(path.name)

        warnings = [
            "Standard image files do not provide reliable CRS/bounds metadata.",
            "Use GeoTIFF for geospatial analysis when available.",
        ]

        if modality == "unknown":
            warnings.append("Sensor/modality could not be identified from the filename.")

        metadata = SceneMetadata(
            filename=path.name,
            format=img.format or path.suffix.lstrip(".").upper(),
            modality=modality,
            modality_source=msource,
            modality_confidence=round(mconf, 3),
            sensor=sensor,
            sensor_source=ssource,
            width=int(img.width),
            height=int(img.height),
            band_count=int(data.shape[0]),
            dtype=str(data.dtype),
            crs=None,
            bounds=None,
            resolution=None,
            acquisition_time=None,
            nodata=None,
            is_georeferenced=False,
            metadata_available=False,
            statistics=_stats(data),
            warnings=warnings,
        )

        profile = {
            "driver": "Pillow",
            "count": int(data.shape[0]),
            "width": int(img.width),
            "height": int(img.height),
            "dtype": str(data.dtype),
            "mode": "RGB",
        }

    return Scene(metadata=metadata, data=data, profile=profile)


def ingest_scene(file_path: str | Path) -> Scene:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Scene file does not exist: {path}")

    suffix = path.suffix.lower()

    if suffix in RASTER_EXTENSIONS:
        return _read_geotiff(path)

    if suffix in IMAGE_EXTENSIONS:
        return _read_standard_image(path)

    raise ValueError(
        f"Unsupported scene format '{suffix}'. "
        f"Supported: {sorted(IMAGE_EXTENSIONS | RASTER_EXTENSIONS)}"
    )


def scene_validation_summary(scene: Scene) -> dict[str, Any]:
    m = scene.metadata

    return {
        "valid": True,
        "filename": m.filename,
        "format": m.format,
        "modality": m.modality,
        "modality_confidence": m.modality_confidence,
        "modality_source": m.modality_source,
        "sensor": m.sensor,
        "sensor_source": m.sensor_source,
        "dimensions": {"width": m.width, "height": m.height},
        "band_count": m.band_count,
        "dtype": m.dtype,
        "crs": m.crs,
        "bounds": m.bounds,
        "resolution": m.resolution,
        "acquisition_time": m.acquisition_time,
        "nodata": m.nodata,
        "is_georeferenced": m.is_georeferenced,
        "metadata_available": m.metadata_available,
        "warnings": m.warnings,
    }
