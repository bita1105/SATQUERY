# ============================================================
# ASTRA GROUNDING ENGINE
# ============================================================

from io import BytesIO
import base64

import cv2
import numpy as np
import torch
from PIL import Image

from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "google/owlv2-base-patch16"

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# Detection settings
DETECTION_THRESHOLD = 0.25
NMS_THRESHOLD = 0.45
MAX_DETECTIONS = 10

# Ignore extremely large regions
MAX_BOX_AREA_RATIO = 0.45

# Ignore extremely tiny regions
MIN_BOX_AREA_RATIO = 0.0002


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("LOADING ASTRA GROUNDING MODEL")
print("=" * 60)

print(
    "Model:",
    MODEL_NAME
)

print(
    "Device:",
    DEVICE
)

processor = AutoProcessor.from_pretrained(
    MODEL_NAME
)

model = AutoModelForZeroShotObjectDetection.from_pretrained(
    MODEL_NAME
)

model.to(DEVICE)
model.eval()

print(
    "ASTRA Grounding model loaded successfully!"
)


# ============================================================
# QUERY -> PRIMARY DETECTION LABEL
# ============================================================

def extract_detection_labels(
    query: str
) -> list[str]:
    """
    Convert the natural-language query into one
    or more canonical grounding concepts.

    We intentionally use a small number of
    primary concepts to reduce duplicate detections.
    """

    q = query.lower().strip()

    # --------------------------------------------------------
    # BUILDINGS
    # --------------------------------------------------------

    if any(
        word in q
        for word in [
            "building",
            "buildings",
            "house",
            "houses",
            "structure",
            "structures",
            "construction",
        ]
    ):
        return [
            "a building"
        ]

    # --------------------------------------------------------
    # ROADS
    # --------------------------------------------------------

    if any(
        word in q
        for word in [
            "road",
            "roads",
            "street",
            "streets",
            "highway",
            "highways",
        ]
    ):
        return [
            "a road"
        ]

    # --------------------------------------------------------
    # WATER
    # --------------------------------------------------------

    if any(
        word in q
        for word in [
            "water",
            "water body",
            "water bodies",
            "river",
            "rivers",
            "lake",
            "lakes",
            "pond",
            "ponds",
        ]
    ):
        return [
            "a water body"
        ]

    # --------------------------------------------------------
    # TREES / VEGETATION
    # --------------------------------------------------------

    if any(
        word in q
        for word in [
            "tree",
            "trees",
            "vegetation",
            "forest",
            "forests",
            "greenery",
        ]
    ):
        return [
            "vegetation"
        ]

    # --------------------------------------------------------
    # VEHICLES
    # --------------------------------------------------------

    if any(
        word in q
        for word in [
            "vehicle",
            "vehicles",
            "car",
            "cars",
            "truck",
            "trucks",
        ]
    ):
        return [
            "a vehicle"
        ]

    # --------------------------------------------------------
    # AIRCRAFT
    # --------------------------------------------------------

    if any(
        word in q
        for word in [
            "aircraft",
            "plane",
            "planes",
            "airplane",
            "airplanes",
        ]
    ):
        return [
            "an airplane"
        ]

    # --------------------------------------------------------
    # AGRICULTURE
    # --------------------------------------------------------

    if any(
        word in q
        for word in [
            "agriculture",
            "agricultural",
            "farmland",
            "farm",
            "crop",
            "crops",
            "field",
            "fields",
        ]
    ):
        return [
            "an agricultural field"
        ]

    # --------------------------------------------------------
    # URBAN
    # --------------------------------------------------------

    if any(
        word in q
        for word in [
            "urban",
            "city",
            "town",
        ]
    ):
        return [
            "a building"
        ]

    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    return [
        "a building"
    ]


# ============================================================
# IMAGE -> BASE64
# ============================================================

def image_to_base64(
    image: Image.Image
) -> str:

    buffer = BytesIO()

    image.save(
        buffer,
        format="PNG"
    )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return (
        "data:image/png;base64,"
        + encoded
    )


# ============================================================
# BOX AREA
# ============================================================

def box_area(
    box,
    image_width: int,
    image_height: int
) -> float:

    x1, y1, x2, y2 = box

    width = max(
        0.0,
        x2 - x1
    )

    height = max(
        0.0,
        y2 - y1
    )

    area = (
        width *
        height
    )

    total_area = (
        image_width *
        image_height
    )

    if total_area <= 0:
        return 0.0

    return (
        area /
        total_area
    )


# ============================================================
# CLIP BOX TO IMAGE
# ============================================================

def clip_box(
    box,
    image_width: int,
    image_height: int
):
    """
    Ensure bounding box coordinates remain inside
    the image boundaries.
    """

    x1, y1, x2, y2 = box

    x1 = max(
        0.0,
        min(
            float(x1),
            image_width - 1
        )
    )

    y1 = max(
        0.0,
        min(
            float(y1),
            image_height - 1
        )
    )

    x2 = max(
        0.0,
        min(
            float(x2),
            image_width - 1
        )
    )

    y2 = max(
        0.0,
        min(
            float(y2),
            image_height - 1
        )
    )

    return [
        x1,
        y1,
        x2,
        y2
    ]


# ============================================================
# NON-MAXIMUM SUPPRESSION
# ============================================================

def apply_nms(
    detections: list[dict]
) -> list[dict]:
    """
    Remove highly overlapping duplicate detections.

    OpenCV NMSBoxes expects boxes in:
    [x, y, width, height]
    format.
    """

    if not detections:
        return []

    boxes = []
    scores = []

    for detection in detections:

        box = detection["box"]

        x1 = int(
            box["x1"]
        )

        y1 = int(
            box["y1"]
        )

        x2 = int(
            box["x2"]
        )

        y2 = int(
            box["y2"]
        )

        width = max(
            1,
            x2 - x1
        )

        height = max(
            1,
            y2 - y1
        )

        boxes.append([
            x1,
            y1,
            width,
            height
        ])

        scores.append(
            float(
                detection["confidence"]
            )
        )

    indices = cv2.dnn.NMSBoxes(
        boxes,
        scores,
        score_threshold=DETECTION_THRESHOLD,
        nms_threshold=NMS_THRESHOLD
    )

    if indices is None:
        return []

    # OpenCV may return [[0], [2]] or [0, 2]
    indices = np.array(
        indices
    ).flatten().tolist()

    kept = [
        detections[index]
        for index in indices
    ]

    kept.sort(
        key=lambda item:
        item["confidence"],
        reverse=True
    )

    return kept


# ============================================================
# GROUNDING
# ============================================================

def analyze_grounding(
    image: Image.Image,
    query: str
) -> dict:
    """
    Perform zero-shot object grounding using OWLv2.
    """

    print("\n" + "=" * 60)
    print("GROUNDING ANALYSIS")
    print("=" * 60)

    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    image_rgb = image.convert(
        "RGB"
    )

    image_width = image_rgb.width
    image_height = image_rgb.height

    # --------------------------------------------------------
    # DETECTION LABELS
    # --------------------------------------------------------

    labels = extract_detection_labels(
        query
    )

    print(
        "Detection labels:",
        labels
    )

    # --------------------------------------------------------
    # PREPARE INPUTS
    # --------------------------------------------------------

    text_labels = [
        labels
    ]

    inputs = processor(
        text=text_labels,
        images=image_rgb,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(DEVICE)
        if hasattr(value, "to")
        else value
        for key, value in inputs.items()
    }

    # --------------------------------------------------------
    # MODEL INFERENCE
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = model(
            **inputs
        )

    # --------------------------------------------------------
    # POST PROCESS
    # --------------------------------------------------------

    target_sizes = torch.tensor(
        [
            [
                image_height,
                image_width
            ]
        ]
    )

    results = (
        processor.post_process_grounded_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=DETECTION_THRESHOLD,
            text_labels=text_labels
        )
    )

    result = results[0]

    boxes = result.get(
        "boxes",
        []
    )

    scores = result.get(
        "scores",
        []
    )

    detected_labels = result.get(
        "text_labels",
        []
    )

    # --------------------------------------------------------
    # RAW DETECTIONS
    # --------------------------------------------------------

    raw_detections = []

    for box, score, label in zip(
        boxes,
        scores,
        detected_labels
    ):

        confidence = float(
            score.item()
        )

        box_values = [
            float(value)
            for value in box.tolist()
        ]

        clipped_box = clip_box(
            box_values,
            image_width,
            image_height
        )

        x1, y1, x2, y2 = (
            clipped_box
        )

        # ----------------------------------------------------
        # Ignore invalid boxes
        # ----------------------------------------------------

        if x2 <= x1 or y2 <= y1:
            continue

        # ----------------------------------------------------
        # Area filtering
        # ----------------------------------------------------

        area_ratio = box_area(
            clipped_box,
            image_width,
            image_height
        )

        # Reject extremely tiny boxes
        if area_ratio < MIN_BOX_AREA_RATIO:
            continue

        # Reject enormous boxes
        if area_ratio > MAX_BOX_AREA_RATIO:
            print(
                "Rejected oversized box:",
                round(
                    area_ratio * 100,
                    2
                ),
                "%"
            )

            continue

        raw_detections.append({

            "label":
                str(label),

            "confidence":
                round(
                    confidence,
                    3
                ),

            "box": {

                "x1":
                    round(x1, 2),

                "y1":
                    round(y1, 2),

                "x2":
                    round(x2, 2),

                "y2":
                    round(y2, 2),
            },

            "area_ratio":
                round(
                    area_ratio,
                    5
                )
        })

    print(
        "Raw valid detections:",
        len(raw_detections)
    )

    # --------------------------------------------------------
    # NMS
    # --------------------------------------------------------

    detections = apply_nms(
        raw_detections
    )

    # --------------------------------------------------------
    # MAX DETECTIONS
    # --------------------------------------------------------

    detections = sorted(
        detections,
        key=lambda item:
        item["confidence"],
        reverse=True
    )

    detections = detections[
        :MAX_DETECTIONS
    ]

    print(
        "Final detections:",
        len(detections)
    )

    for detection in detections:

        print(
            f"  {detection['label']} "
            f"confidence="
            f"{detection['confidence']} "
            f"box="
            f"{detection['box']}"
        )

    # --------------------------------------------------------
    # DRAW GROUNDING MAP
    # --------------------------------------------------------

    image_array = np.array(
        image_rgb
    )

    annotated = cv2.cvtColor(
        image_array,
        cv2.COLOR_RGB2BGR
    )

    for index, detection in enumerate(
        detections,
        start=1
    ):

        box = detection[
            "box"
        ]

        x1 = int(
            box["x1"]
        )

        y1 = int(
            box["y1"]
        )

        x2 = int(
            box["x2"]
        )

        y2 = int(
            box["y2"]
        )

        # ----------------------------------------------------
        # Bounding box
        # ----------------------------------------------------

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            (255, 255, 0),
            3
        )

        # ----------------------------------------------------
        # Label
        # ----------------------------------------------------

        label_text = (
            f"{detection['label']} "
            f"{detection['confidence']:.2f}"
        )

        text_y = max(
            25,
            y1 - 8
        )

        cv2.putText(
            annotated,
            label_text,
            (
                x1,
                text_y
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 0),
            2,
            cv2.LINE_AA
        )

        # ----------------------------------------------------
        # Detection index
        # ----------------------------------------------------

        cv2.putText(
            annotated,
            f"#{index}",
            (
                x1 + 5,
                y1 + 22
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 0),
            2,
            cv2.LINE_AA
        )

    # --------------------------------------------------------
    # Convert to PIL
    # --------------------------------------------------------

    annotated_rgb = cv2.cvtColor(
        annotated,
        cv2.COLOR_BGR2RGB
    )

    annotated_image = Image.fromarray(
        annotated_rgb
    )

    # --------------------------------------------------------
    # Base64
    # --------------------------------------------------------

    grounding_map = image_to_base64(
        annotated_image
    )

    # --------------------------------------------------------
    # Detection statistics
    # --------------------------------------------------------

    if detections:

        average_confidence = (
            sum(
                detection[
                    "confidence"
                ]
                for detection in detections
            )
            /
            len(detections)
        )

    else:

        average_confidence = 0.0

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    return {

        "query":
            query,

        "labels_searched":
            labels,

        "detections":
            detections,

        "detection_count":
            len(detections),

        "average_confidence":
            round(
                float(
                    average_confidence
                ),
                3
            ),

        "grounding_map":
            grounding_map,

        "image_size": {

            "width":
                image_width,

            "height":
                image_height,
        },

        "parameters": {

            "detection_threshold":
                DETECTION_THRESHOLD,

            "nms_threshold":
                NMS_THRESHOLD,

            "max_detections":
                MAX_DETECTIONS,

            "max_box_area_ratio":
                MAX_BOX_AREA_RATIO,

            "min_box_area_ratio":
                MIN_BOX_AREA_RATIO,
        }
    }