from io import BytesIO
import base64
import os
import tempfile

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageChops, ImageEnhance

from vqa_engine import analyze_image
from answer_engine import build_answer
from agent_controller import classify_query
from change_vqa import analyze_change_vqa
from grounding_engine import analyze_grounding
from optical_sar_engine import analyze_optical_sar
from scene_ingestion import ingest_scene, scene_validation_summary


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="SatQuery AI API",
    description="Agentic remote-sensing intelligence backend",
    version="0.6.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://satquery-psi.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return {
        "system": "SatQuery AI",
        "status": "operational",
        "message": (
            "Remote-sensing intelligence backend is running."
        ),
        "pipeline": (
            "input_validation -> "
            "agent_controller -> "
            "specialist_tool -> "
            "evidence -> "
            "answer"
        ),
    }


# ============================================================
# IMAGE VALIDATION
# ============================================================

def validate_image(image_bytes: bytes):
    max_file_size = 20 * 1024 * 1024

    if len(image_bytes) > max_file_size:
        raise ValueError(
            "Image file is too large. "
            "Maximum allowed size is 20 MB."
        )

    try:
        original = Image.open(BytesIO(image_bytes))
        original.verify()

        img = Image.open(
            BytesIO(image_bytes)
        ).convert("RGB")

        image_format = Image.open(
            BytesIO(image_bytes)
        ).format

    except Exception:
        raise ValueError(
            "The uploaded file is not a valid image."
        )

    supported_formats = {
        "JPEG",
        "PNG",
        "TIFF",
        "WEBP",
    }

    if image_format not in supported_formats:
        raise ValueError(
            f"Unsupported image format: {image_format}. "
            f"Supported formats: "
            f"{', '.join(sorted(supported_formats))}"
        )

    return img, image_format


# ============================================================
# GEOSPATIAL SCENE INGESTION
# ============================================================

async def ingest_uploaded_scene(
    upload: UploadFile,
    raw_bytes: bytes,
):
    """
    Persist the original upload temporarily so Rasterio can inspect GeoTIFF
    metadata while PIL continues to support ordinary image formats.
    """

    suffix = os.path.splitext(
        upload.filename or ""
    )[1].lower() or ".bin"

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        ) as temp_file:

            temp_file.write(raw_bytes)
            temp_path = temp_file.name

        scene = ingest_scene(temp_path)

        return scene

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def build_scene_pair_validation(
    primary_scene,
    secondary_scene=None,
):
    """
    Creates a compact scene-validation block for API/UI consumption.
    """

    result = {
        "primary": scene_validation_summary(
            primary_scene
        ),
    }

    if secondary_scene is not None:
        result["secondary"] = scene_validation_summary(
            secondary_scene
        )

    # Cross-scene compatibility checks.
    compatibility = {
        "same_dimensions": False,
        "same_crs": False,
        "same_bounds": False,
        "same_modality": False,
        "same_sensor": False,
        "compatible_for_optical_sar": False,
        "warnings": [],
    }

    if secondary_scene is not None:

        p = primary_scene.metadata
        s = secondary_scene.metadata

        compatibility["same_dimensions"] = (
            p.width == s.width and
            p.height == s.height
        )

        compatibility["same_crs"] = (
            p.crs is not None and
            s.crs is not None and
            p.crs == s.crs
        )

        compatibility["same_bounds"] = (
            p.bounds is not None and
            s.bounds is not None and
            p.bounds == s.bounds
        )

        compatibility["same_modality"] = (
            p.modality != "unknown"
            and
            p.modality == s.modality
        )

        compatibility["same_sensor"] = (
            p.sensor != "unknown"
            and
            p.sensor == s.sensor
        )

        compatibility["compatible_for_optical_sar"] = (
            {p.modality, s.modality} == {"optical", "sar"}
        )

        if p.modality == "unknown":
            compatibility["warnings"].append(
                "Primary scene modality could not be identified."
            )

        if s.modality == "unknown":
            compatibility["warnings"].append(
                "Secondary scene modality could not be identified."
            )

        if (
            p.crs is not None
            and
            s.crs is not None
            and
            p.crs != s.crs
        ):
            compatibility["warnings"].append(
                "The two scenes use different CRS values. "
                "Reprojection will be required before geospatial fusion."
            )

        if (
            p.bounds is not None
            and
            s.bounds is not None
            and
            p.bounds != s.bounds
        ):
            compatibility["warnings"].append(
                "The two scenes have different spatial bounds."
            )

        if (
            p.modality != "unknown"
            and
            s.modality != "unknown"
            and
            {p.modality, s.modality} != {"optical", "sar"}
        ):
            compatibility["warnings"].append(
                "The two uploaded scenes are not an Optical + SAR pair."
            )

    result["compatibility"] = compatibility

    return result


# ============================================================
# SIMPLE CHANGE MAP
# ============================================================

def generate_change_map(
    before: Image.Image,
    after: Image.Image,
):
    target_size = (
        min(before.width, after.width),
        min(before.height, after.height),
    )

    before_resized = before.resize(target_size)
    after_resized = after.resize(target_size)

    difference = ImageChops.difference(
        before_resized,
        after_resized,
    )

    difference = ImageEnhance.Contrast(
        difference
    ).enhance(3.0)

    gray = difference.convert("L")

    threshold = 30

    binary = gray.point(
        lambda p: 255 if p > threshold else 0
    )

    changed_pixels = sum(
        1
        for pixel in binary.getdata()
        if pixel > 0
    )

    total_pixels = (
        binary.width *
        binary.height
    )

    unchanged_pixels = (
        total_pixels -
        changed_pixels
    )

    changed_percentage = (
        changed_pixels /
        total_pixels *
        100
    )

    unchanged_percentage = (
        unchanged_pixels /
        total_pixels *
        100
    )

    change_map = Image.new(
        "RGB",
        binary.size,
        (0, 0, 0),
    )

    pixels = change_map.load()
    binary_pixels = binary.load()

    for y in range(binary.height):
        for x in range(binary.width):
            if binary_pixels[x, y] > 0:
                pixels[x, y] = (
                    255,
                    70,
                    70,
                )
            else:
                pixels[x, y] = (
                    5,
                    8,
                    12,
                )

    buffer = BytesIO()

    change_map.save(
        buffer,
        format="PNG",
    )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return {
        "changed_area_percent": round(
            changed_percentage,
            2,
        ),
        "unchanged_area_percent": round(
            unchanged_percentage,
            2,
        ),
        "changed_pixels": changed_pixels,
        "total_pixels": total_pixels,
        "change_map": (
            "data:image/png;base64," +
            encoded
        ),
    }


# ============================================================
# ANALYZE IMAGE
# ============================================================

@app.post("/api/analyze")
async def analyze(
    query: str = Form(...),
    image: UploadFile = File(...),
    second_image: UploadFile | None = File(
        default=None
    ),
):
    print("\n" + "=" * 70)
    print("SATQUERY ANALYSIS REQUEST")
    print("=" * 70)

    print("Query:", query)
    print("Primary image:", image.filename)

    if second_image:
        print(
            "Secondary image:",
            second_image.filename,
        )

    # ========================================================
    # 1. PRIMARY IMAGE
    # ========================================================

    if not image.filename:
        return {
            "status": "error",
            "message": (
                "No primary image was provided."
            ),
        }

    try:
        image_bytes = await image.read()

        img, image_format = validate_image(
            image_bytes
        )

        primary_scene = await ingest_uploaded_scene(
            image,
            image_bytes,
        )

    except ValueError as e:
        return {
            "status": "error",
            "message": str(e),
        }

    # ========================================================
    # 2. SECOND IMAGE
    # ========================================================

    second_img = None
    second_format = None
    second_bytes = None
    secondary_scene = None

    if second_image:
        if not second_image.filename:
            return {
                "status": "error",
                "message": (
                    "Secondary image was provided "
                    "without a filename."
                ),
            }

        try:
            second_bytes = await second_image.read()

            second_img, second_format = validate_image(
                second_bytes
            )

            secondary_scene = await ingest_uploaded_scene(
                second_image,
                second_bytes,
            )

        except ValueError as e:
            return {
                "status": "error",
                "message": (
                    f"Secondary image error: {str(e)}"
                ),
            }

    # ========================================================
    # 3. AGENT CONTROLLER
    # ========================================================

    try:
        routing = classify_query(
            query,
            second_img is not None,
        )

    except TypeError:
        routing = classify_query(query)

    except Exception as e:
        return {
            "status": "error",
            "message": (
                "Agent controller failed."
            ),
            "details": str(e),
        }

    task = routing["task"]

    reason = routing.get(
        "reason",
        "Task classified by agent controller.",
    )

    requires_second_image = routing.get(
        "requires_second_image",
        False,
    )

    print("\nAGENT DECISION")
    print("Task:", task)
    print("Reason:", reason)

    # ========================================================
    # 4. SCENE VALIDATION / GEOSPATIAL INGESTION
    # ========================================================

    scene_pair = build_scene_pair_validation(
        primary_scene,
        secondary_scene,
    )

    print("\nSCENE INGESTION")
    print(
        "Primary modality:",
        primary_scene.metadata.modality,
        "| Sensor:",
        primary_scene.metadata.sensor,
    )

    if secondary_scene is not None:
        print(
            "Secondary modality:",
            secondary_scene.metadata.modality,
            "| Sensor:",
            secondary_scene.metadata.sensor,
        )

    print(
        "Primary georeferenced:",
        primary_scene.metadata.is_georeferenced,
    )

    if secondary_scene is not None:
        print(
            "Optical-SAR compatible by metadata:",
            scene_pair["compatibility"][
                "compatible_for_optical_sar"
            ],
        )

    # ========================================================
    # 5. SECOND IMAGE REQUIREMENT
    # ========================================================

    if requires_second_image and second_img is None:
        return {
            "status": "error",
            "query": query,
            "detected_task": task,
            "confidence": 0.0,
            "message": (
                f"The agent identified '{task}', "
                "which requires two images. "
                "Please upload both primary and "
                "secondary imagery."
            ),
            "agent": {
                "task": task,
                "reason": reason,
                "requires_second_image": True,
            },
            "execution": [
                "Input validated",
                "Query received",
                "Task classification",
                "Required secondary image missing",
            ],
        }

    # ========================================================
    # 6. VQA
    # ========================================================

    if task == "vqa":
        try:
            print("\nRunning VQA specialist...")

            analysis = analyze_image(img)
            evidence = analysis["evidence"]

            answer = build_answer(
                query,
                evidence,
            )

        except Exception as e:
            print("VQA ERROR:", str(e))

            return {
                "status": "error",
                "message": (
                    "The VQA specialist failed."
                ),
                "details": str(e),
            }

        return {
            "status": "success",
            "query": query,
            "image": {
                "filename": image.filename,
                "content_type": image.content_type,
                "format": image_format,
                "width": img.width,
                "height": img.height,
                "mode": img.mode,
                "size_bytes": len(image_bytes),
                "size_mb": round(
                    len(image_bytes) /
                    (1024 * 1024),
                    2,
                ),
            },
            "validation": {
                "is_valid": True,
                "format_supported": True,
            },
            "scene": scene_pair,
            "detected_task": "vqa",
            "confidence": 0.94,
            "evidence": evidence,
            "answer": answer,
            "message": answer,
            "agent": {
                "task": task,
                "reason": reason,
                "requires_second_image": False,
            },
            "execution": [
                "Input validated",
                "Scene metadata ingested",
                "Query received",
                "Task classification",
                "VQA specialist executed",
                "Evidence extracted",
                "Answer generated",
                "Result returned",
            ],
        }

    # ========================================================
    # 7. CHANGE DETECTION
    # ========================================================

    if task == "change_detection":
        try:
            print(
                "\nRunning Change Detection specialist..."
            )

            result = generate_change_map(
                img,
                second_img,
            )

            changed_percentage = result[
                "changed_area_percent"
            ]

            answer = (
                "Visual comparison indicates "
                f"approximately {changed_percentage}% "
                "of the compared image area contains "
                "detectable visual differences. "
                "The current prototype identifies "
                "where visual change occurred, but "
                "does not semantically classify the "
                "specific type of change."
            )

        except Exception as e:
            print(
                "CHANGE DETECTION ERROR:",
                str(e),
            )

            return {
                "status": "error",
                "message": (
                    "The change detection specialist "
                    "failed."
                ),
                "details": str(e),
            }

        return {
            "status": "success",
            "query": query,
            "detected_task": "change_detection",
            "scene": scene_pair,
            "confidence": 0.88,
            "message": answer,
            "answer": answer,
            "change_detection": result,
            "change_map": result[
                "change_map"
            ],
            "changed_area_percent": result[
                "changed_area_percent"
            ],
            "unchanged_area_percent": result[
                "unchanged_area_percent"
            ],
            "changed_pixels": result[
                "changed_pixels"
            ],
            "total_pixels": result[
                "total_pixels"
            ],
            "agent": {
                "task": task,
                "reason": reason,
                "requires_second_image": True,
            },
            "execution": [
                "Input validated",
                "Scene metadata ingested",
                "Query received",
                "Task classification",
                "Change detection executed",
                "Pixel difference calculated",
                "Change map generated",
                "Change statistics calculated",
                "Result returned",
            ],
        }

    # ========================================================
    # 8. CHANGE VQA
    # ========================================================

    if task == "change_vqa":
        try:
            print(
                "\nRunning Change VQA specialist..."
            )

            result = analyze_change_vqa(
                img,
                second_img,
                query,
            )

            answer = result["answer"]

        except Exception as e:
            print(
                "CHANGE VQA ERROR:",
                str(e),
            )

            return {
                "status": "error",
                "message": (
                    "The Change VQA specialist failed."
                ),
                "details": str(e),
            }

        return {
            "status": "success",
            "query": query,
            "detected_task": "change_vqa",
            "scene": scene_pair,
            "confidence": 0.86,
            "message": answer,
            "answer": answer,

            "before_evidence":
                result["before_evidence"],

            "after_evidence":
                result["after_evidence"],

            "changes":
                result["changes"],

            "changed_pixels":
                result["changed_pixels"],

            "total_pixels":
                result["total_pixels"],

            "changed_area_percent":
                result["changed_area_percent"],

            "unchanged_area_percent":
                result["unchanged_area_percent"],

            "change_map":
                result["change_map"],

            "registration":
                result.get("registration"),

            "detection":
                result.get("detection"),

            "change_vqa": {
                "before_evidence":
                    result["before_evidence"],

                "after_evidence":
                    result["after_evidence"],

                "changes":
                    result["changes"],

                "changed_area_percent":
                    result["changed_area_percent"],

                "unchanged_area_percent":
                    result["unchanged_area_percent"],

                "registration":
                    result.get("registration"),

                "detection":
                    result.get("detection"),
            },

            "agent": {
                "task": task,
                "reason": reason,
                "requires_second_image": True,
            },

            "execution": [
                "Input validated",
                "Scene metadata ingested",
                "Query received",
                "Task classification",
                "Change VQA specialist selected",
                "Before-image evidence extracted",
                "After-image evidence extracted",
                "Semantic evidence compared",
                "Image registration performed",
                "Pixel-level visual difference calculated",
                "Change map generated",
                "Change statistics calculated",
                "Change interpretation generated",
                "Result returned",
            ],
        }

    # ========================================================
    # 9. GROUNDING
    # ========================================================

    if task == "grounding":
        try:
            print(
                "\nRunning Grounding specialist..."
            )

            result = analyze_grounding(
                img,
                query,
            )

            detection_count = result[
                "detection_count"
            ]

            average_confidence = result.get(
                "average_confidence",
                0.0,
            )

            if detection_count > 0:
                answer = (
                    f"Grounding identified "
                    f"{detection_count} candidate "
                    "regions matching the requested "
                    "visual concept. Bounding boxes "
                    "are shown in the grounding map."
                )
            else:
                answer = (
                    "No candidate regions were detected "
                    "for the requested visual concept."
                )

        except Exception as e:
            print(
                "GROUNDING ERROR:",
                str(e),
            )

            return {
                "status": "error",
                "message": (
                    "The grounding specialist failed."
                ),
                "details": str(e),
            }

        return {
            "status": "success",
            "query": query,
            "detected_task": "grounding",
            "scene": scene_pair,
            "confidence": average_confidence,
            "message": answer,
            "answer": answer,

            "grounding": result,

            "grounding_map":
                result["grounding_map"],

            "detections":
                result["detections"],

            "detection_count":
                result["detection_count"],

            "labels_searched":
                result["labels_searched"],

            "average_confidence":
                result.get(
                    "average_confidence",
                    0.0,
                ),

            "parameters":
                result.get(
                    "parameters",
                    {},
                ),

            "agent": {
                "task": task,
                "reason": reason,
                "requires_second_image": False,
            },

            "execution": [
                "Input validated",
                "Scene metadata ingested",
                "Query received",
                "Task classification",
                "Grounding specialist selected",
                "Natural-language detection labels generated",
                "Zero-shot object grounding executed",
                "Candidate regions detected",
                "Bounding boxes generated",
                "Grounding map generated",
                "Result returned",
            ],
        }

    # ========================================================
    # 10. OPTICAL + SAR
    # ========================================================

    if task == "optical_sar_analysis":
        try:
            print(
                "\nRunning Optical + SAR specialist..."
            )

            if second_img is None:
                return {
                    "status": "error",
                    "query": query,
                    "detected_task":
                        "optical_sar_analysis",
                    "confidence": 0.0,
                    "message": (
                        "Optical + SAR analysis "
                        "requires two images: "
                        "one optical image and "
                        "one SAR image."
                    ),
                    "agent": {
                        "task": task,
                        "reason": reason,
                        "requires_second_image": True,
                    },
                    "execution": [
                        "Input validated",
                        "Query received",
                        "Task classification",
                        "Optical-SAR secondary image missing",
                    ],
                }

            print(
                "Optical image:",
                image.filename,
            )

            print(
                "SAR image:",
                second_image.filename,
            )

            optical_sar_result = analyze_optical_sar(
                img,
                second_img,
            )

            confidence = optical_sar_result.get(
                "confidence",
                0.0,
            )

            answer = optical_sar_result.get(
                "answer",
                (
                    "Optical + SAR analysis "
                    "completed."
                ),
            )

            print(
                "Optical + SAR confidence:",
                confidence,
            )

            print(
                "Registration:",
                optical_sar_result.get(
                    "registration",
                    {}
                ).get(
                    "status",
                    "unknown",
                ),
            )

        except Exception as e:
            print(
                "OPTICAL-SAR ERROR:",
                str(e),
            )

            return {
                "status": "error",
                "query": query,
                "detected_task":
                    "optical_sar_analysis",
                "confidence": 0.0,
                "message": (
                    "The Optical + SAR specialist "
                    "failed."
                ),
                "details": str(e),
                "agent": {
                    "task": task,
                    "reason": reason,
                    "requires_second_image": True,
                },
                "execution": [
                    "Input validated",
                    "Query received",
                    "Task classification",
                    "Optical-SAR specialist failed",
                ],
            }

        return {
            "status": "success",
            "query": query,
            "detected_task":
                "optical_sar_analysis",

            "scene":
                scene_pair,

            "confidence":
                confidence,

            "message":
                answer,

            "answer":
                answer,

            "optical_sar":
                optical_sar_result,

            "confidence_type":
                optical_sar_result.get(
                    "confidence_type",
                    "baseline_heuristic",
                ),

            "confidence_label":
                optical_sar_result.get(
                    "confidence_label",
                    "Baseline Fusion Score",
                ),

            "reliability":
                optical_sar_result.get(
                    "reliability",
                    "normal",
                ),

            "warnings":
                optical_sar_result.get(
                    "warnings",
                    [],
                ),

            "registration":
                optical_sar_result.get(
                    "registration"
                ),

            "cross_modal_findings":
                optical_sar_result.get(
                    "cross_modal_findings",
                    [],
                ),

            "optical":
                optical_sar_result.get(
                    "optical",
                    {},
                ),

            "sar":
                optical_sar_result.get(
                    "sar",
                    {},
                ),

            "image_size":
                optical_sar_result.get(
                    "image_size",
                    {},
                ),

            "agent": {
                "task": task,
                "reason": reason,
                "requires_second_image": True,
            },

            "execution": [
                "Input validated",
                "Scene metadata ingested",
                "Query received",
                "Task classification",
                "Optical image accepted",
                "SAR image accepted",
                "Optical-SAR registration check",
                "Optical features extracted",
                "SAR features extracted",
                "Cross-modal fusion completed",
                "Confidence estimated",
                "Joint answer generated",
                "Result returned",
            ],
        }

    # ========================================================
    # 11. UNKNOWN TASK
    # ========================================================

    return {
        "status": "error",
        "query": query,
        "detected_task": task,
        "confidence": 0.0,
        "message": (
            "The agent identified an unsupported "
            "analysis task."
        ),
        "agent": {
            "task": task,
            "reason": reason,
            "requires_second_image":
                requires_second_image,
        },
        "execution": [
            "Input validated",
            "Query received",
            "Task classification",
            "Unsupported task",
        ],
    }
