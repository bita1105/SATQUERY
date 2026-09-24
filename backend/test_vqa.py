from PIL import Image

from vqa_engine import analyze_image
from answer_engine import build_answer


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_PATH = r"C:\Users\proph\OneDrive\Desktop\testimage.png"


# ============================================================
# LOAD IMAGE
# ============================================================

print("Loading image...")

image = Image.open(
    IMAGE_PATH
).convert("RGB")

print("Image loaded!")


# ============================================================
# RUN SATQUERY VQA PIPELINE
# ============================================================

print("\n" + "=" * 60)
print("RUNNING SATQUERY VISUAL ANALYSIS")
print("=" * 60)

result = analyze_image(image)

evidence = result["evidence"]


# ============================================================
# DISPLAY RAW EVIDENCE
# ============================================================

print("\n" + "=" * 60)
print("RAW VISUAL EVIDENCE")
print("=" * 60)

for key, value in evidence.items():
    print(f"{key}: {value}")


# ============================================================
# SATQUERY ANSWER ENGINE
# ============================================================

queries = [
    "Describe this image",
    "What is in this image?",
    "What are the main land cover types?",
    "Are there any water bodies?",
    "Is there vegetation?",
    "Are there buildings?",
    "Are there roads?",
]


print("\n" + "=" * 60)
print("SATQUERY ANSWER ENGINE")
print("=" * 60)


for query in queries:

    answer = build_answer(
        query,
        evidence
    )

    print("\nQuery:", query)
    print("SatQuery:", answer)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)