def classify_query(query: str, has_second_image: bool = False) -> dict:
    q = query.lower().strip()

    # ============================================================
    # OPTICAL + SAR
    # ============================================================

    multimodal_keywords = [
        "sar",
        "optical and sar",
        "optical + sar",
        "optical-sar",
        "optical sar",
        "multispectral and sar",
        "combine sar",
        "radar and optical",
        "optical and radar",
        "multimodal",
        "cross-modal",
        "cross modal",
    ]

    is_optical_sar_query = any(
        keyword in q for keyword in multimodal_keywords
    )

    if is_optical_sar_query:
        return {
            "task": "optical_sar_analysis",
            "reason": (
                "The query requests complementary analysis of "
                "optical and SAR/radar imagery."
            ),
            "requires_second_image": True,
        }

    # ============================================================
    # CHANGE / TEMPORAL
    # ============================================================

    change_keywords = [
        "what changed",
        "change",
        "changes",
        "changed",
        "difference",
        "differences",
        "before and after",
        "compare",
        "comparison",
        "temporal",
        "over time",
        "between the images",
        "between these images",
    ]

    if has_second_image or any(
        keyword in q for keyword in change_keywords
    ):
        task = (
            "change_vqa"
            if has_second_image
            else "change_detection"
        )

        return {
            "task": task,
            "reason": (
                "The query indicates temporal or comparative analysis."
            ),
            "requires_second_image": True,
        }

    # ============================================================
    # GROUNDING
    # ============================================================

    grounding_keywords = [
        "where",
        "locate",
        "location of",
        "highlight",
        "show me where",
        "point out",
        "identify the location",
        "mark",
        "bounding box",
    ]

    if any(keyword in q for keyword in grounding_keywords):
        return {
            "task": "grounding",
            "reason": (
                "The query asks to locate or highlight "
                "an object or region."
            ),
            "requires_second_image": False,
        }

    # ============================================================
    # VQA
    # ============================================================

    vqa_keywords = [
        "what is in the image",
        "what does the image show",
        "describe the image",
        "describe this image",
        "identify",
        "is there",
        "are there",
        "how many",
        "what type",
        "what kind",
        "is this",
        "does the image contain",
    ]

    if any(keyword in q for keyword in vqa_keywords):
        return {
            "task": "vqa",
            "reason": (
                "The query asks a visual question about "
                "a single image."
            ),
            "requires_second_image": False,
        }

    # ============================================================
    # DEFAULT
    # ============================================================

    return {
        "task": "vqa",
        "reason": (
            "No specialized task was explicitly detected; "
            "defaulting to visual question answering."
        ),
        "requires_second_image": False,
    }