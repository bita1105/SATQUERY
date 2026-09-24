from PIL import Image, ImageChops, ImageEnhance, ImageFilter
import numpy as np
import base64
from io import BytesIO


# ============================================================
# CONFIGURATION
# ============================================================

CHANGE_THRESHOLD = 35
BLUR_RADIUS = 2
MIN_IMAGE_SIZE = 256


# ============================================================
# IMAGE PREPARATION
# ============================================================

def prepare_image(image: Image.Image) -> Image.Image:
    """
    Convert image to RGB and resize it to a common size.
    """

    image = image.convert("RGB")

    width, height = image.size

    if width < MIN_IMAGE_SIZE or height < MIN_IMAGE_SIZE:
        image = image.resize(
            (MIN_IMAGE_SIZE, MIN_IMAGE_SIZE)
        )

    return image


def align_images(
    image_a: Image.Image,
    image_b: Image.Image
):
    """
    Resize both images to the same dimensions.

    NOTE:
    This is a basic alignment step.
    Proper geospatial registration will be added later.
    """

    image_a = prepare_image(image_a)
    image_b = prepare_image(image_b)

    target_width = min(
        image_a.width,
        image_b.width
    )

    target_height = min(
        image_a.height,
        image_b.height
    )

    image_a = image_a.resize(
        (target_width, target_height)
    )

    image_b = image_b.resize(
        (target_width, target_height)
    )

    return image_a, image_b


# ============================================================
# CHANGE CALCULATION
# ============================================================

def calculate_change_mask(
    image_a: Image.Image,
    image_b: Image.Image
):
    """
    Calculate pixel-level visual difference
    between two images.
    """

    # Slight blur reduces sensitivity to tiny noise
    image_a_blur = image_a.filter(
        ImageFilter.GaussianBlur(BLUR_RADIUS)
    )

    image_b_blur = image_b.filter(
        ImageFilter.GaussianBlur(BLUR_RADIUS)
    )

    # Absolute difference
    difference = ImageChops.difference(
        image_a_blur,
        image_b_blur
    )

    # Convert difference to grayscale
    grayscale_difference = difference.convert(
        "L"
    )

    diff_array = np.array(
        grayscale_difference
    )

    # Threshold
    change_mask = (
        diff_array > CHANGE_THRESHOLD
    )

    return difference, change_mask


# ============================================================
# CHANGE STATISTICS
# ============================================================

def calculate_statistics(
    change_mask: np.ndarray
):
    """
    Calculate basic change statistics.
    """

    total_pixels = change_mask.size

    changed_pixels = int(
        np.sum(change_mask)
    )

    changed_percentage = (
        changed_pixels /
        total_pixels
    ) * 100

    unchanged_percentage = (
        100 -
        changed_percentage
    )

    return {
        "total_pixels": total_pixels,
        "changed_pixels": changed_pixels,
        "changed_percentage": round(
            changed_percentage,
            2
        ),
        "unchanged_percentage": round(
            unchanged_percentage,
            2
        )
    }


# ============================================================
# GENERATE CHANGE MAP
# ============================================================

def generate_change_map(
    difference: Image.Image,
    change_mask: np.ndarray
):
    """
    Generate a visual change map.

    Changed pixels are highlighted while
    unchanged regions remain dark.
    """

    # Enhance difference for visualization
    enhanced = ImageEnhance.Contrast(
        difference.convert("RGB")
    ).enhance(2.5)

    enhanced_array = np.array(
        enhanced
    )

    # Darken everything first
    result = (
        enhanced_array * 0.20
    ).astype(np.uint8)

    # Highlight changed pixels
    result[change_mask] = [
        255,
        80,
        80
    ]

    return Image.fromarray(
        result
    )


# ============================================================
# IMAGE → BASE64
# ============================================================

def image_to_base64(
    image: Image.Image
):
    """
    Convert PIL image into a base64 data URL
    so the frontend can display it directly.
    """

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
# MAIN CHANGE DETECTION
# ============================================================

def detect_changes(
    image_a: Image.Image,
    image_b: Image.Image
):
    """
    Complete change-detection pipeline.
    """

    # --------------------------------------------------------
    # 1. Align / resize
    # --------------------------------------------------------

    image_a, image_b = align_images(
        image_a,
        image_b
    )

    # --------------------------------------------------------
    # 2. Calculate difference
    # --------------------------------------------------------

    difference, change_mask = (
        calculate_change_mask(
            image_a,
            image_b
        )
    )

    # --------------------------------------------------------
    # 3. Statistics
    # --------------------------------------------------------

    statistics = calculate_statistics(
        change_mask
    )

    # --------------------------------------------------------
    # 4. Change map
    # --------------------------------------------------------

    change_map = generate_change_map(
        difference,
        change_mask
    )

    # --------------------------------------------------------
    # 5. Base64 output
    # --------------------------------------------------------

    change_map_base64 = image_to_base64(
        change_map
    )

    # --------------------------------------------------------
    # 6. Return
    # --------------------------------------------------------

    return {
        "statistics": statistics,
        "change_map": change_map_base64,
        "image_size": {
            "width": image_a.width,
            "height": image_a.height
        }
    }