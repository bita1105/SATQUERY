from PIL import Image, ImageChops, ImageFilter
from io import BytesIO
import base64

import cv2
import numpy as np

from vqa_engine import ask_vqa


# ============================================================
# CHANGE VQA CONFIGURATION
# ============================================================

CHANGE_QUESTIONS = {
    "vegetation": "Is there vegetation in the image?",
    "water": "Are there any water bodies?",
    "buildings": "Are there buildings visible in the image?",
    "roads": "Are there roads visible in the image?",
    "agriculture": "Are there agricultural fields in the image?",
    "urban": "Is there a dense urban area in the image?",
}


# ============================================================
# NORMALIZE VQA ANSWER
# ============================================================

def normalize_answer(answer: str) -> str:

    answer = answer.strip().lower()

    if answer in {
        "yes",
        "yeah",
        "yep",
        "true",
        "present",
        "visible",
    }:
        return "yes"

    if answer in {
        "no",
        "nope",
        "false",
        "absent",
        "not visible",
    }:
        return "no"

    return "uncertain"


# ============================================================
# ANALYZE ONE IMAGE
# ============================================================

def analyze_single_image(image: Image.Image) -> dict:

    evidence = {}

    for key, question in CHANGE_QUESTIONS.items():

        answer = ask_vqa(
            image,
            question
        )

        evidence[key] = normalize_answer(answer)

    return evidence


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
# PIL -> OPENCV
# ============================================================

def pil_to_cv2(
    image: Image.Image
) -> np.ndarray:

    rgb = np.array(
        image.convert("RGB")
    )

    return cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2BGR
    )


# ============================================================
# OPENCV -> PIL
# ============================================================

def cv2_to_pil(
    image: np.ndarray
) -> Image.Image:

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    return Image.fromarray(rgb)


# ============================================================
# IMAGE REGISTRATION
# ============================================================

def register_images(
    before_image: Image.Image,
    after_image: Image.Image
) -> dict:
    """
    Align the AFTER image to the BEFORE image using
    ORB feature matching and homography.

    Returns the aligned AFTER image along with
    registration statistics and a valid overlap mask.
    """

    print("\n" + "-" * 60)
    print("IMAGE REGISTRATION")
    print("-" * 60)

    before_cv = pil_to_cv2(
        before_image
    )

    after_cv = pil_to_cv2(
        after_image
    )

    # --------------------------------------------------------
    # Make AFTER the same dimensions as BEFORE
    # --------------------------------------------------------

    target_width = before_cv.shape[1]
    target_height = before_cv.shape[0]

    after_resized = cv2.resize(
        after_cv,
        (
            target_width,
            target_height
        ),
        interpolation=cv2.INTER_AREA
    )

    # --------------------------------------------------------
    # Create working-size images
    # --------------------------------------------------------

    max_dimension = 1200

    scale = min(
        1.0,
        max_dimension /
        max(
            target_width,
            target_height
        )
    )

    working_width = max(
        1,
        int(target_width * scale)
    )

    working_height = max(
        1,
        int(target_height * scale)
    )

    before_work = cv2.resize(
        before_cv,
        (
            working_width,
            working_height
        ),
        interpolation=cv2.INTER_AREA
    )

    after_work = cv2.resize(
        after_resized,
        (
            working_width,
            working_height
        ),
        interpolation=cv2.INTER_AREA
    )

    # --------------------------------------------------------
    # Grayscale
    # --------------------------------------------------------

    before_gray = cv2.cvtColor(
        before_work,
        cv2.COLOR_BGR2GRAY
    )

    after_gray = cv2.cvtColor(
        after_work,
        cv2.COLOR_BGR2GRAY
    )

    # Slight contrast normalization
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    before_gray = clahe.apply(
        before_gray
    )

    after_gray = clahe.apply(
        after_gray
    )

    # --------------------------------------------------------
    # ORB FEATURE DETECTION
    # --------------------------------------------------------

    orb = cv2.ORB_create(
        nfeatures=2500,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=31,
        fastThreshold=15
    )

    keypoints_before, descriptors_before = (
        orb.detectAndCompute(
            before_gray,
            None
        )
    )

    keypoints_after, descriptors_after = (
        orb.detectAndCompute(
            after_gray,
            None
        )
    )

    print(
        "BEFORE keypoints:",
        len(keypoints_before)
    )

    print(
        "AFTER keypoints:",
        len(keypoints_after)
    )

    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    if (
        descriptors_before is None
        or descriptors_after is None
        or len(keypoints_before) < 8
        or len(keypoints_after) < 8
    ):

        print(
            "Registration could not find enough features."
        )

        print(
            "Using resized AFTER image without "
            "geometric registration."
        )

        valid_mask = np.ones(
            (
                target_height,
                target_width
            ),
            dtype=np.uint8
        ) * 255

        return {
            "aligned_after": after_resized,

            "valid_mask": valid_mask,

            "registration_success": False,

            "good_matches": 0,

            "inliers": 0,

            "inlier_ratio": 0.0,

            "registration_method":
                "resize_fallback",
        }

    # --------------------------------------------------------
    # MATCH DESCRIPTORS
    # --------------------------------------------------------

    matcher = cv2.BFMatcher(
        cv2.NORM_HAMMING
    )

    matches = matcher.knnMatch(
        descriptors_after,
        descriptors_before,
        k=2
    )

    # --------------------------------------------------------
    # LOWE RATIO TEST
    # --------------------------------------------------------

    good_matches = []

    for pair in matches:

        if len(pair) < 2:
            continue

        m, n = pair

        if m.distance < 0.75 * n.distance:

            good_matches.append(m)

    print(
        "Good feature matches:",
        len(good_matches)
    )

    # --------------------------------------------------------
    # HOMOGRAPHY
    # --------------------------------------------------------

    if len(good_matches) < 8:

        print(
            "Not enough good matches for homography."
        )

        valid_mask = np.ones(
            (
                target_height,
                target_width
            ),
            dtype=np.uint8
        ) * 255

        return {
            "aligned_after": after_resized,

            "valid_mask": valid_mask,

            "registration_success": False,

            "good_matches":
                len(good_matches),

            "inliers": 0,

            "inlier_ratio": 0.0,

            "registration_method":
                "resize_fallback",
        }

    src_pts = np.float32([
        keypoints_after[m.queryIdx].pt
        for m in good_matches
    ]).reshape(-1, 1, 2)

    dst_pts = np.float32([
        keypoints_before[m.trainIdx].pt
        for m in good_matches
    ]).reshape(-1, 1, 2)

    try:

        H, mask = cv2.findHomography(
            src_pts,
            dst_pts,
            cv2.RANSAC,
            5.0
        )

    except cv2.error as e:

        print(
            "Homography calculation failed:",
            str(e)
        )

        H = None
        mask = None

    # --------------------------------------------------------
    # HOMOGRAPHY VALIDATION
    # --------------------------------------------------------

    if H is None or mask is None:

        print(
            "Valid homography was not obtained."
        )

        valid_mask = np.ones(
            (
                target_height,
                target_width
            ),
            dtype=np.uint8
        ) * 255

        return {
            "aligned_after": after_resized,

            "valid_mask": valid_mask,

            "registration_success": False,

            "good_matches":
                len(good_matches),

            "inliers": 0,

            "inlier_ratio": 0.0,

            "registration_method":
                "resize_fallback",
        }

    inliers = int(
        mask.ravel().sum()
    )

    inlier_ratio = (
        inliers /
        max(len(good_matches), 1)
    )

    print(
        "Homography inliers:",
        inliers
    )

    print(
        "Inlier ratio:",
        round(
            inlier_ratio,
            3
        )
    )

    # --------------------------------------------------------
    # Reject weak registration
    # --------------------------------------------------------

    if (
        inliers < 8
        or inlier_ratio < 0.20
    ):

        print(
            "Registration confidence is too low."
        )

        print(
            "Using resized AFTER image as fallback."
        )

        valid_mask = np.ones(
            (
                target_height,
                target_width
            ),
            dtype=np.uint8
        ) * 255

        return {
            "aligned_after": after_resized,

            "valid_mask": valid_mask,

            "registration_success": False,

            "good_matches":
                len(good_matches),

            "inliers":
                inliers,

            "inlier_ratio":
                round(
                    float(inlier_ratio),
                    3
                ),

            "registration_method":
                "resize_fallback",
        }

    # --------------------------------------------------------
    # Convert small-image homography to full resolution
    # --------------------------------------------------------

    scale_matrix = np.array([
        [scale, 0, 0],
        [0, scale, 0],
        [0, 0, 1]
    ], dtype=np.float64)

    inverse_scale_matrix = np.linalg.inv(
        scale_matrix
    )

    H_full = (
        inverse_scale_matrix
        @ H
        @ scale_matrix
    )

    # --------------------------------------------------------
    # Warp AFTER image
    # --------------------------------------------------------

    aligned_after = cv2.warpPerspective(
        after_resized,
        H_full,
        (
            target_width,
            target_height
        ),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )

    # --------------------------------------------------------
    # Create valid overlap mask
    # --------------------------------------------------------

    source_mask = np.ones(
        (
            target_height,
            target_width
        ),
        dtype=np.uint8
    ) * 255

    valid_mask = cv2.warpPerspective(
        source_mask,
        H_full,
        (
            target_width,
            target_height
        ),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    # Remove a small edge region
    kernel = np.ones(
        (5, 5),
        dtype=np.uint8
    )

    valid_mask = cv2.erode(
        valid_mask,
        kernel,
        iterations=1
    )

    print(
        "Image registration successful."
    )

    return {
        "aligned_after":
            aligned_after,

        "valid_mask":
            valid_mask,

        "registration_success":
            True,

        "good_matches":
            len(good_matches),

        "inliers":
            inliers,

        "inlier_ratio":
            round(
                float(inlier_ratio),
                3
            ),

        "registration_method":
            "ORB + RANSAC homography",
    }


# ============================================================
# REGISTERED PIXEL-LEVEL CHANGE ANALYSIS
# ============================================================

# ============================================================
# ROBUST REGISTERED CHANGE DETECTION
# ============================================================

def calculate_pixel_change(
    before_image: Image.Image,
    after_image: Image.Image
) -> dict:
    """
    Perform registered change detection using:
    1. Image registration
    2. Colour normalization
    3. Local structural difference
    4. Adaptive thresholding
    5. Morphological filtering
    6. Connected-component filtering
    """

    print("\n" + "-" * 60)
    print("ROBUST CHANGE DETECTION")
    print("-" * 60)

    # --------------------------------------------------------
    # 1. REGISTER IMAGES
    # --------------------------------------------------------

    registration = register_images(
        before_image,
        after_image
    )

    before_cv = pil_to_cv2(
        before_image
    )

    aligned_after = registration[
        "aligned_after"
    ]

    valid_mask = registration[
        "valid_mask"
    ]

    # --------------------------------------------------------
    # 2. NORMALIZE IMAGE SIZE
    # --------------------------------------------------------

    height, width = before_cv.shape[:2]

    if aligned_after.shape[:2] != (
        height,
        width
    ):

        aligned_after = cv2.resize(
            aligned_after,
            (width, height),
            interpolation=cv2.INTER_AREA
        )

    # --------------------------------------------------------
    # 3. CONVERT TO LAB
    # --------------------------------------------------------

    before_lab = cv2.cvtColor(
        before_cv,
        cv2.COLOR_BGR2LAB
    )

    after_lab = cv2.cvtColor(
        aligned_after,
        cv2.COLOR_BGR2LAB
    )

    # --------------------------------------------------------
    # 4. CLAHE ON LUMINANCE
    # --------------------------------------------------------

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    before_lab[:, :, 0] = clahe.apply(
        before_lab[:, :, 0]
    )

    after_lab[:, :, 0] = clahe.apply(
        after_lab[:, :, 0]
    )

    # --------------------------------------------------------
    # 5. COLOUR NORMALIZATION
    # --------------------------------------------------------

    before_float = (
        before_lab.astype(
            np.float32
        )
    )

    after_float = (
        after_lab.astype(
            np.float32
        )
    )

    # Normalize each channel around its mean
    for channel in range(3):

        before_channel = (
            before_float[:, :, channel]
        )

        after_channel = (
            after_float[:, :, channel]
        )

        before_mean = (
            np.mean(before_channel[
                valid_mask > 0
            ])
            if np.any(valid_mask > 0)
            else np.mean(before_channel)
        )

        after_mean = (
            np.mean(after_channel[
                valid_mask > 0
            ])
            if np.any(valid_mask > 0)
            else np.mean(after_channel)
        )

        before_std = (
            np.std(before_channel[
                valid_mask > 0
            ])
            if np.any(valid_mask > 0)
            else np.std(before_channel)
        )

        after_std = (
            np.std(after_channel[
                valid_mask > 0
            ])
            if np.any(valid_mask > 0)
            else np.std(after_channel)
        )

        before_std = max(
            before_std,
            1.0
        )

        after_std = max(
            after_std,
            1.0
        )

        before_float[:, :, channel] = (
            (
                before_channel -
                before_mean
            ) /
            before_std
        )

        after_float[:, :, channel] = (
            (
                after_channel -
                after_mean
            ) /
            after_std
        )

    # --------------------------------------------------------
    # 6. CHANNEL DIFFERENCE
    # --------------------------------------------------------

    channel_difference = np.mean(
        np.abs(
            before_float -
            after_float
        ),
        axis=2
    )

    # --------------------------------------------------------
    # 7. LOCAL BLUR
    # --------------------------------------------------------

    before_gray = cv2.cvtColor(
        before_cv,
        cv2.COLOR_BGR2GRAY
    )

    after_gray = cv2.cvtColor(
        aligned_after,
        cv2.COLOR_BGR2GRAY
    )

    before_gray = cv2.GaussianBlur(
        before_gray,
        (7, 7),
        0
    )

    after_gray = cv2.GaussianBlur(
        after_gray,
        (7, 7),
        0
    )

    structural_difference = cv2.absdiff(
        before_gray,
        after_gray
    ).astype(
        np.float32
    )

    # Normalize structural difference
    structural_difference = (
        structural_difference /
        255.0
    )

    # --------------------------------------------------------
    # 8. COMBINE DIFFERENCE SIGNALS
    # --------------------------------------------------------

    channel_difference = (
        channel_difference /
        max(
            np.percentile(
                channel_difference[
                    valid_mask > 0
                ],
                95
            )
            if np.any(valid_mask > 0)
            else 1.0,
            1e-6
        )
    )

    channel_difference = np.clip(
        channel_difference,
        0,
        1
    )

    combined_difference = (
        0.65 *
        channel_difference
        +
        0.35 *
        structural_difference
    )

    # --------------------------------------------------------
    # 9. IGNORE INVALID REGISTRATION AREAS
    # --------------------------------------------------------

    combined_difference[
        valid_mask == 0
    ] = 0

    # --------------------------------------------------------
    # 10. ADAPTIVE THRESHOLD
    # --------------------------------------------------------

    valid_values = (
        combined_difference[
            valid_mask > 0
        ]
    )

    if len(valid_values) > 0:

        # Keep only strongest differences
        adaptive_threshold = max(
            0.30,
            float(
                np.percentile(
                    valid_values,
                    82
                )
            )
        )

    else:

        adaptive_threshold = 0.30

    print(
        "Adaptive threshold:",
        round(
            adaptive_threshold,
            4
        )
    )

    change_mask = (
        combined_difference >
        adaptive_threshold
    ).astype(
        np.uint8
    ) * 255

    # --------------------------------------------------------
    # 11. REMOVE INVALID AREAS
    # --------------------------------------------------------

    change_mask[
        valid_mask == 0
    ] = 0

    # --------------------------------------------------------
    # 12. MORPHOLOGICAL CLEANING
    # --------------------------------------------------------

    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (9, 9)
    )

    change_mask = cv2.morphologyEx(
        change_mask,
        cv2.MORPH_OPEN,
        open_kernel
    )

    change_mask = cv2.morphologyEx(
        change_mask,
        cv2.MORPH_CLOSE,
        close_kernel
    )

    # --------------------------------------------------------
    # 13. CONNECTED COMPONENT FILTERING
    # --------------------------------------------------------

    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            change_mask,
            connectivity=8
        )
    )

    cleaned_mask = np.zeros_like(
        change_mask
    )

    minimum_component_area = max(
        100,
        int(
            (width * height) *
            0.00005
        )
    )

    for label in range(
        1,
        num_labels
    ):

        area = stats[
            label,
            cv2.CC_STAT_AREA
        ]

        if area >= minimum_component_area:

            cleaned_mask[
                labels == label
            ] = 255

    change_mask = cleaned_mask

    # --------------------------------------------------------
    # 14. STATISTICS
    # --------------------------------------------------------

    valid_pixels = (
        valid_mask > 0
    )

    changed_pixels = int(
        np.sum(
            (
                change_mask > 0
            )
            &
            valid_pixels
        )
    )

    total_pixels = int(
        np.sum(
            valid_pixels
        )
    )

    unchanged_pixels = (
        total_pixels -
        changed_pixels
    )

    if total_pixels > 0:

        changed_percentage = (
            changed_pixels /
            total_pixels
        ) * 100

    else:

        changed_percentage = 0.0

    unchanged_percentage = (
        100 -
        changed_percentage
    )

    # --------------------------------------------------------
    # 15. BUILD CHANGE MAP
    # --------------------------------------------------------

    change_map = before_cv.copy()

    # Slight darkening for presentation
    change_map = cv2.convertScaleAbs(
        change_map,
        alpha=0.85,
        beta=0
    )

    # Red overlay
    red_overlay = np.zeros_like(
        change_map
    )

    red_overlay[:, :] = (
        0,
        0,
        255
    )

    blended = cv2.addWeighted(
        change_map,
        0.60,
        red_overlay,
        0.40,
        0
    )

    changed_region = (
        change_mask > 0
    )

    change_map[
        changed_region
    ] = blended[
        changed_region
    ]

    # --------------------------------------------------------
    # 16. OUTSIDE REGISTRATION OVERLAP
    # --------------------------------------------------------

    outside_overlap = (
        valid_mask == 0
    )

    change_map[
        outside_overlap
    ] = (
        change_map[
            outside_overlap
        ] * 0.20
    ).astype(
        np.uint8
    )

    # --------------------------------------------------------
    # 17. CONVERT TO BASE64
    # --------------------------------------------------------

    change_map_pil = cv2_to_pil(
        change_map
    )

    change_map_base64 = (
        image_to_base64(
            change_map_pil
        )
    )

    # --------------------------------------------------------
    # 18. RETURN
    # --------------------------------------------------------

    return {

        "changed_pixels":
            changed_pixels,

        "total_pixels":
            total_pixels,

        "changed_area_percent":
            round(
                float(
                    changed_percentage
                ),
                2
            ),

        "unchanged_area_percent":
            round(
                float(
                    unchanged_percentage
                ),
                2
            ),

        "change_map":
            change_map_base64,

        "registration": {

            "success":
                registration[
                    "registration_success"
                ],

            "method":
                registration[
                    "registration_method"
                ],

            "good_matches":
                registration[
                    "good_matches"
                ],

            "inliers":
                registration[
                    "inliers"
                ],

            "inlier_ratio":
                registration[
                    "inlier_ratio"
                ],
        },

        "detection": {

            "method":
                "LAB + structural difference",

            "adaptive_threshold":
                round(
                    float(
                        adaptive_threshold
                    ),
                    4
                ),

            "minimum_component_area":
                minimum_component_area,
        }
    }

    registration = register_images(
        before_image,
        after_image
    )

    before_cv = pil_to_cv2(
        before_image
    )

    aligned_after = registration[
        "aligned_after"
    ]

    valid_mask = registration[
        "valid_mask"
    ]

    # --------------------------------------------------------
    # Convert both images to LAB
    # --------------------------------------------------------

    before_lab = cv2.cvtColor(
        before_cv,
        cv2.COLOR_BGR2LAB
    )

    after_lab = cv2.cvtColor(
        aligned_after,
        cv2.COLOR_BGR2LAB
    )

    # Compare luminance channel
    before_l = before_lab[:, :, 0]

    after_l = after_lab[:, :, 0]

    # Smooth small noise
    before_l = cv2.GaussianBlur(
        before_l,
        (5, 5),
        0
    )

    after_l = cv2.GaussianBlur(
        after_l,
        (5, 5),
        0
    )

    # --------------------------------------------------------
    # Absolute difference
    # --------------------------------------------------------

    difference = cv2.absdiff(
        before_l,
        after_l
    )

    # --------------------------------------------------------
    # Threshold
    # --------------------------------------------------------

    threshold = 25

    change_mask = (
        difference > threshold
    ).astype(
        np.uint8
    ) * 255

    # --------------------------------------------------------
    # Ignore areas outside valid overlap
    # --------------------------------------------------------

    change_mask[
        valid_mask == 0
    ] = 0

    # --------------------------------------------------------
    # Remove small noise
    # --------------------------------------------------------

    kernel_open = np.ones(
        (3, 3),
        dtype=np.uint8
    )

    kernel_close = np.ones(
        (5, 5),
        dtype=np.uint8
    )

    change_mask = cv2.morphologyEx(
        change_mask,
        cv2.MORPH_OPEN,
        kernel_open
    )

    change_mask = cv2.morphologyEx(
        change_mask,
        cv2.MORPH_CLOSE,
        kernel_close
    )

    # --------------------------------------------------------
    # Remove tiny connected components
    # --------------------------------------------------------

    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            change_mask,
            connectivity=8
        )
    )

    cleaned_mask = np.zeros_like(
        change_mask
    )

    minimum_component_area = 40

    for label in range(
        1,
        num_labels
    ):

        area = stats[
            label,
            cv2.CC_STAT_AREA
        ]

        if area >= minimum_component_area:

            cleaned_mask[
                labels == label
            ] = 255

    change_mask = cleaned_mask

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    valid_pixels = (
        valid_mask > 0
    )

    changed_pixels = np.sum(
        (
            change_mask > 0
        )
        &
        valid_pixels
    )

    total_pixels = np.sum(
        valid_pixels
    )

    unchanged_pixels = (
        total_pixels -
        changed_pixels
    )

    if total_pixels > 0:

        changed_percentage = (
            changed_pixels /
            total_pixels
        ) * 100

    else:

        changed_percentage = 0.0

    unchanged_percentage = (
        100 -
        changed_percentage
    )

    # --------------------------------------------------------
    # Create interpretable change map
    # --------------------------------------------------------

    change_map = before_cv.copy()

    red_overlay = np.zeros_like(
        before_cv
    )

    red_overlay[:, :] = (
        0,
        0,
        255
    )

    blended = cv2.addWeighted(
        before_cv,
        0.60,
        red_overlay,
        0.40,
        0
    )

    changed_region = (
        change_mask > 0
    )

    change_map[
        changed_region
    ] = blended[
        changed_region
    ]

    # Outside overlap -> darkened
    outside_overlap = (
        valid_mask == 0
    )

    change_map[
        outside_overlap
    ] = (
        change_map[
            outside_overlap
        ] * 0.25
    ).astype(
        np.uint8
    )

    # Convert to PIL
    change_map_pil = cv2_to_pil(
        change_map
    )

    # Base64
    change_map_base64 = (
        image_to_base64(
            change_map_pil
        )
    )

    return {

        "changed_pixels":
            int(changed_pixels),

        "total_pixels":
            int(total_pixels),

        "changed_area_percent":
            round(
                float(changed_percentage),
                2
            ),

        "unchanged_area_percent":
            round(
                float(unchanged_percentage),
                2
            ),

        "change_map":
            change_map_base64,

        "registration": {

            "success":
                registration[
                    "registration_success"
                ],

            "method":
                registration[
                    "registration_method"
                ],

            "good_matches":
                registration[
                    "good_matches"
                ],

            "inliers":
                registration[
                    "inliers"
                ],

            "inlier_ratio":
                registration[
                    "inlier_ratio"
                ],
        }
    }


# ============================================================
# COMPARE SEMANTIC EVIDENCE
# ============================================================

def compare_evidence(
    before: dict,
    after: dict
) -> dict:

    changes = {}

    for key in CHANGE_QUESTIONS.keys():

        before_state = before.get(
            key,
            "uncertain"
        )

        after_state = after.get(
            key,
            "uncertain"
        )

        if (
            before_state == "yes"
            and after_state == "no"
        ):

            changes[key] = {
                "before":
                    before_state,

                "after":
                    after_state,

                "change":
                    "decreased"
            }

        elif (
            before_state == "no"
            and after_state == "yes"
        ):

            changes[key] = {
                "before":
                    before_state,

                "after":
                    after_state,

                "change":
                    "increased"
            }

        elif (
            before_state == after_state
            and before_state in {
                "yes",
                "no"
            }
        ):

            changes[key] = {
                "before":
                    before_state,

                "after":
                    after_state,

                "change":
                    "stable"
            }

        else:

            changes[key] = {
                "before":
                    before_state,

                "after":
                    after_state,

                "change":
                    "uncertain"
            }

    return changes


# ============================================================
# BUILD CHANGE ANSWER
# ============================================================

def build_change_answer(
    query: str,
    changes: dict,
    pixel_change: dict
) -> str:

    query_lower = query.lower()

    changed_percent = (
        pixel_change[
            "changed_area_percent"
        ]
    )

    registration = (
        pixel_change[
            "registration"
        ]
    )

    registration_note = ""

    if registration["success"]:

        registration_note = (
            " The images were spatially aligned "
            "before calculating visual differences."
        )

    else:

        registration_note = (
            " Automatic spatial alignment was not "
            "reliable, so the result should be "
            "treated as a raw visual comparison."
        )

    # --------------------------------------------------------
    # VEGETATION
    # --------------------------------------------------------

    if (
        "vegetation" in query_lower
        or "trees" in query_lower
    ):

        result = changes[
            "vegetation"
        ]

        if result["change"] == "increased":

            return (
                "The semantic evidence indicates an "
                "increase in visible vegetation between "
                "the two images."
                + registration_note
            )

        if result["change"] == "decreased":

            return (
                "The semantic evidence indicates a "
                "decrease in visible vegetation between "
                "the two images."
                + registration_note
            )

        if result["change"] == "stable":

            return (
                "Vegetation is detected in both images. "
                f"Registered pixel-level analysis shows "
                f"approximately {changed_percent}% "
                "of the overlapping image area "
                "underwent measurable visual change."
                + registration_note
            )

        return (
            "The available visual evidence is insufficient "
            "to determine the vegetation change reliably."
        )


    # --------------------------------------------------------
    # WATER
    # --------------------------------------------------------

    if any(
        word in query_lower
        for word in [
            "water",
            "river",
            "lake",
            "sea",
            "ocean",
        ]
    ):

        result = changes[
            "water"
        ]

        if result["change"] == "increased":

            return (
                "The semantic evidence indicates an "
                "increase in visible water presence."
                + registration_note
            )

        if result["change"] == "decreased":

            return (
                "The semantic evidence indicates a "
                "decrease in visible water presence."
                + registration_note
            )

        if result["change"] == "stable":

            return (
                "Water is detected consistently in both "
                "images. Registered pixel-level analysis "
                f"detected visual change across "
                f"approximately {changed_percent}% "
                "of the overlapping image area."
                + registration_note
            )

        return (
            "The available evidence is insufficient to "
            "determine the change in water presence."
        )


    # --------------------------------------------------------
    # BUILDINGS / URBAN
    # --------------------------------------------------------

    if any(
        word in query_lower
        for word in [
            "building",
            "buildings",
            "structure",
            "urban",
            "city",
            "town",
            "development",
        ]
    ):

        building_change = (
            changes["buildings"]
        )

        urban_change = (
            changes["urban"]
        )

        if (
            building_change["change"]
            == "increased"
            or
            urban_change["change"]
            == "increased"
        ):

            return (
                "The semantic evidence indicates an "
                "increase in visible built-up or urban "
                "features."
                + registration_note
            )

        if (
            building_change["change"]
            == "decreased"
            or
            urban_change["change"]
            == "decreased"
        ):

            return (
                "The semantic evidence indicates a "
                "decrease in visible built-up or urban "
                "features."
                + registration_note
            )

        return (
            "Built-up or urban features show no clear "
            "presence/absence change between the two "
            "images. Registered pixel-level analysis "
            f"detected visual change across "
            f"approximately {changed_percent}% "
            "of the overlapping image area."
            + registration_note
        )


    # --------------------------------------------------------
    # AGRICULTURE
    # --------------------------------------------------------

    if any(
        word in query_lower
        for word in [
            "agriculture",
            "agricultural",
            "farmland",
            "farm",
            "crop",
            "crops",
            "fields",
        ]
    ):

        result = changes[
            "agriculture"
        ]

        if result["change"] == "increased":

            return (
                "The semantic evidence indicates an "
                "increase in visible agricultural areas."
                + registration_note
            )

        if result["change"] == "decreased":

            return (
                "The semantic evidence indicates a "
                "decrease in visible agricultural areas."
                + registration_note
            )

        return (
            "Agricultural presence shows no clear "
            "presence/absence change. Registered "
            "pixel-level analysis detected visual "
            f"change across approximately "
            f"{changed_percent}% of the overlapping "
            "image area."
            + registration_note
        )


    # --------------------------------------------------------
    # GENERAL CHANGE QUESTION
    # --------------------------------------------------------

    if any(
        word in query_lower
        for word in [
            "what changed",
            "changes",
            "difference",
            "different",
            "changed between",
            "compare",
            "comparison",
        ]
    ):

        increased = []
        decreased = []
        stable = []

        for key, result in changes.items():

            if result["change"] == "increased":

                increased.append(key)

            elif result["change"] == "decreased":

                decreased.append(key)

            elif result["change"] == "stable":

                stable.append(key)

        statements = []

        if increased:

            statements.append(
                "increased visibility of "
                + ", ".join(increased)
            )

        if decreased:

            statements.append(
                "decreased visibility of "
                + ", ".join(decreased)
            )

        if stable:

            statements.append(
                "stable semantic presence of "
                + ", ".join(stable)
            )

        if statements:

            summary = "; ".join(
                statements
            )

            return (
                "The two images show "
                + summary
                + ". Registered pixel-level "
                "comparison indicates approximately "
                f"{changed_percent}% of the overlapping "
                "image area underwent measurable visual "
                "change."
                + registration_note
            )

        return (
            "The two images show measurable visual "
            f"difference across approximately "
            f"{changed_percent}% of the overlapping "
            "image area, but the current semantic "
            "evidence is insufficient to identify "
            "the specific land-cover change reliably."
            + registration_note
        )


    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    return (
        "The two images were compared using semantic "
        "visual evidence and registered pixel-level "
        f"difference analysis. Approximately "
        f"{changed_percent}% of the overlapping "
        "image area shows measurable visual change."
        + registration_note
    )


# ============================================================
# MAIN CHANGE VQA PIPELINE
# ============================================================

def analyze_change_vqa(
    before_image: Image.Image,
    after_image: Image.Image,
    query: str
) -> dict:

    print("\n" + "=" * 60)
    print("CHANGE VQA ANALYSIS")
    print("=" * 60)

    # --------------------------------------------------------
    # BEFORE IMAGE
    # --------------------------------------------------------

    print(
        "Analyzing BEFORE image..."
    )

    before_evidence = (
        analyze_single_image(
            before_image
        )
    )

    print(
        "BEFORE evidence:"
    )

    for key, value in (
        before_evidence.items()
    ):

        print(
            f"  {key}: {value}"
        )


    # --------------------------------------------------------
    # AFTER IMAGE
    # --------------------------------------------------------

    print(
        "\nAnalyzing AFTER image..."
    )

    after_evidence = (
        analyze_single_image(
            after_image
        )
    )

    print(
        "AFTER evidence:"
    )

    for key, value in (
        after_evidence.items()
    ):

        print(
            f"  {key}: {value}"
        )


    # --------------------------------------------------------
    # SEMANTIC COMPARISON
    # --------------------------------------------------------

    print(
        "\nComparing semantic evidence..."
    )

    changes = compare_evidence(
        before_evidence,
        after_evidence
    )

    for key, value in (
        changes.items()
    ):

        print(
            f"  {key}: "
            f"{value['before']} -> "
            f"{value['after']} "
            f"({value['change']})"
        )


    # --------------------------------------------------------
    # REGISTERED PIXEL COMPARISON
    # --------------------------------------------------------

    print(
        "\nCalculating registered "
        "pixel-level change..."
    )

    pixel_change = (
        calculate_pixel_change(
            before_image,
            after_image
        )
    )

    print(
        "Registration successful:",
        pixel_change[
            "registration"
        ]["success"]
    )

    print(
        "Registration method:",
        pixel_change[
            "registration"
        ]["method"]
    )

    print(
        "Good matches:",
        pixel_change[
            "registration"
        ]["good_matches"]
    )

    print(
        "Inliers:",
        pixel_change[
            "registration"
        ]["inliers"]
    )

    print(
        "Inlier ratio:",
        pixel_change[
            "registration"
        ]["inlier_ratio"]
    )

    print(
        "Changed area:",
        pixel_change[
            "changed_area_percent"
        ],
        "%"
    )

    print(
        "Unchanged area:",
        pixel_change[
            "unchanged_area_percent"
        ],
        "%"
    )


    # --------------------------------------------------------
    # FINAL ANSWER
    # --------------------------------------------------------

    answer = build_change_answer(
        query,
        changes,
        pixel_change
    )

    print(
        "\nFINAL CHANGE VQA ANSWER:"
    )

    print(answer)


    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    return {

        "before_evidence":
            before_evidence,

        "after_evidence":
            after_evidence,

        "changes":
            changes,

        "changed_pixels":
            pixel_change[
                "changed_pixels"
            ],

        "total_pixels":
            pixel_change[
                "total_pixels"
            ],

        "changed_area_percent":
            pixel_change[
                "changed_area_percent"
            ],

        "unchanged_area_percent":
            pixel_change[
                "unchanged_area_percent"
            ],

        "change_map":
            pixel_change[
                "change_map"
            ],

        "registration":
            pixel_change[
                "registration"
            ],

        "answer":
            answer,
    }