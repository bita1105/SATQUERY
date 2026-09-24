from PIL import Image
from transformers import AutoProcessor, BlipForQuestionAnswering


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "Salesforce/blip-vqa-base"


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading SatQuery VQA model...")

processor = AutoProcessor.from_pretrained(MODEL_NAME)

model = BlipForQuestionAnswering.from_pretrained(MODEL_NAME)

print("SatQuery VQA model loaded successfully!")


# ============================================================
# TARGETED REMOTE-SENSING QUESTIONS
# ============================================================

EVIDENCE_QUESTIONS = {
    "vegetation": "Is there vegetation in the image?",
    "water": "Are there any water bodies?",
    "buildings": "Are there buildings visible in the image?",
    "roads": "Are there roads visible in the image?",
    "agriculture": "Are there agricultural fields in the image?",
    "forest": "Are there forests or dense trees in the image?",
    "urban": "Is there a dense urban area in the image?",
    "bare_land": "Is there bare or exposed land in the image?",
}


# ============================================================
# ASK VQA MODEL
# ============================================================

def ask_vqa(image: Image.Image, question: str) -> str:
    """
    Ask BLIP a question about an image.
    """

    inputs = processor(
        images=image,
        text=question,
        return_tensors="pt"
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=20
    )

    answer = processor.decode(
        outputs[0],
        skip_special_tokens=True
    ).strip().lower()

    return answer


# ============================================================
# COLLECT TARGETED VISUAL EVIDENCE
# ============================================================

def collect_evidence(image: Image.Image) -> dict:
    """
    Ask a fixed set of remote-sensing questions and
    return the resulting visual evidence.
    """

    evidence = {}

    for key, question in EVIDENCE_QUESTIONS.items():

        answer = ask_vqa(
            image,
            question
        )

        evidence[key] = answer

    return evidence


# ============================================================
# ANALYZE IMAGE
# ============================================================

def analyze_image(image: Image.Image) -> dict:
    """
    Main VQA pipeline used by the SatQuery backend.
    """

    evidence = collect_evidence(image)

    return {
        "evidence": evidence
    }