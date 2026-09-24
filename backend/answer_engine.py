def normalize_answer(answer: str) -> str:
    """
    Convert raw VQA output into a simple evidence state.
    """

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


def build_answer(query: str, evidence: dict) -> str:

    query_lower = query.lower()

    # --------------------------------------------------
    # NORMALIZE BOOLEAN EVIDENCE
    # --------------------------------------------------

    vegetation = normalize_answer(
        evidence.get("vegetation", "")
    )

    water = normalize_answer(
        evidence.get("water", "")
    )

    buildings = normalize_answer(
        evidence.get("buildings", "")
    )

    roads = normalize_answer(
        evidence.get("roads", "")
    )

    agriculture = normalize_answer(
        evidence.get("agriculture", "")
    )

    urban = normalize_answer(
        evidence.get("urban", "")
    )

    # These are descriptive rather than simple yes/no.
    forest = evidence.get("forest", "uncertain")
    bare_land = evidence.get("bare_land", "uncertain")

    # --------------------------------------------------
    # WATER
    # --------------------------------------------------

    if any(word in query_lower for word in [
        "water",
        "river",
        "lake",
        "sea",
        "ocean",
        "water bodies",
    ]):

        if water == "yes":
            return (
                "Yes. Water bodies or water channels are "
                "visible in the imagery."
            )

        if water == "no":
            return (
                "No obvious water bodies were detected "
                "by the visual analysis."
            )

        return (
            "The presence of water bodies could not be "
            "determined reliably."
        )

    # --------------------------------------------------
    # VEGETATION
    # --------------------------------------------------

    if any(word in query_lower for word in [
        "vegetation",
        "trees",
        "tree",
        "greenery",
    ]):

        if vegetation == "yes":
            return (
                "Yes. Vegetation is visibly present across "
                "the scene."
            )

        if vegetation == "no":
            return (
                "No significant vegetation was detected "
                "by the visual analysis."
            )

        return (
            "The presence of vegetation could not be "
            "determined reliably."
        )

    # --------------------------------------------------
    # BUILDINGS
    # --------------------------------------------------

    if any(word in query_lower for word in [
        "building",
        "buildings",
        "structure",
        "structures",
    ]):

        if buildings == "yes":
            return (
                "Yes. Buildings or man-made structures "
                "appear to be visible in the imagery."
            )

        if buildings == "no":
            return (
                "No obvious buildings or structures were "
                "detected by the visual analysis."
            )

        return (
            "The presence of buildings could not be "
            "determined reliably."
        )

    # --------------------------------------------------
    # ROADS
    # --------------------------------------------------

    if any(word in query_lower for word in [
        "road",
        "roads",
        "highway",
        "street",
    ]):

        if roads == "yes":
            return (
                "Yes. Road-like linear features appear "
                "to be visible in the imagery."
            )

        if roads == "no":
            return (
                "No obvious roads were detected by the "
                "visual analysis."
            )

        return (
            "The presence of roads could not be "
            "determined reliably."
        )

    # --------------------------------------------------
    # AGRICULTURE
    # --------------------------------------------------

    if any(word in query_lower for word in [
        "agriculture",
        "agricultural",
        "farmland",
        "farm",
        "farms",
        "crop",
        "crops",
        "fields",
    ]):

        if agriculture == "yes":
            return (
                "Agricultural or cultivated areas appear "
                "to be present in the imagery."
            )

        if agriculture == "no":
            return (
                "No clear agricultural fields were "
                "detected by the visual analysis."
            )

        return (
            "The presence of agricultural land could "
            "not be determined reliably."
        )

    # --------------------------------------------------
    # URBAN
    # --------------------------------------------------

    if any(word in query_lower for word in [
        "urban",
        "city",
        "town",
        "developed area",
        "development",
    ]):

        if urban == "yes":
            return (
                "The visual analysis indicates the "
                "presence of a potentially developed or "
                "urbanized area."
            )

        if urban == "no":
            return (
                "No dense urban development was detected "
                "by the visual analysis."
            )

        return (
            "The degree of urban development could not "
            "be determined reliably."
        )

    # --------------------------------------------------
    # FOREST
    # --------------------------------------------------

    if any(word in query_lower for word in [
        "forest",
        "forested",
        "dense trees",
    ]):

        if forest not in ["", "uncertain"]:
            return (
                f"The model describes the tree/forest "
                f"presence as '{forest}'."
            )

        return (
            "The presence of forest or dense tree cover "
            "could not be determined reliably."
        )

    # --------------------------------------------------
    # BARE LAND
    # --------------------------------------------------

    if any(word in query_lower for word in [
        "bare land",
        "bare ground",
        "exposed land",
        "exposed ground",
    ]):

        if bare_land not in ["", "uncertain"]:
            return (
                f"The model describes exposed or bare "
                f"land as '{bare_land}'."
            )

        return (
            "The presence of bare or exposed land could "
            "not be determined reliably."
        )

    # --------------------------------------------------
    # LAND COVER
    # --------------------------------------------------

    if any(word in query_lower for word in [
        "land cover",
        "landcover",
        "cover types",
    ]):

        parts = []

        if vegetation == "yes":
            parts.append("vegetated areas")

        if water == "yes":
            parts.append("water bodies")

        if agriculture == "yes":
            parts.append("agricultural areas")

        if urban == "yes":
            parts.append("developed/urban areas")

        if bare_land not in ["", "uncertain"]:
            parts.append("bare or exposed land")

        if parts:
            return (
                "The available visual evidence indicates "
                "the presence of "
                + ", ".join(parts)
                + "."
            )

        return (
            "The main land-cover types could not be "
            "determined reliably."
        )

    # --------------------------------------------------
    # GENERAL DESCRIPTION
    # --------------------------------------------------

    if any(word in query_lower for word in [
        "describe",
        "what is in",
        "what do you see",
        "what does this image show",
    ]):

        parts = []

        if vegetation == "yes":
            parts.append("visible vegetation")

        if water == "yes":
            parts.append("water bodies")

        if buildings == "yes":
            parts.append("buildings or structures")

        if roads == "yes":
            parts.append("road-like features")

        if agriculture == "yes":
            parts.append("agricultural areas")

        if urban == "yes":
            parts.append("developed or urbanized areas")

        if bare_land not in ["", "uncertain"]:
            parts.append("bare or exposed land")

        if parts:

            if len(parts) == 1:
                description = parts[0]

            elif len(parts) == 2:
                description = f"{parts[0]} and {parts[1]}"

            else:
                description = (
                    ", ".join(parts[:-1])
                    + ", and "
                    + parts[-1]
                )

            return (
                "The imagery shows "
                + description
                + ". These observations are based on "
                  "the current visual analysis."
            )

        return (
            "The image could not be described reliably "
            "from the available visual evidence."
        )

    # --------------------------------------------------
    # FALLBACK
    # --------------------------------------------------

    return (
        "The current visual analysis does not provide "
        "enough evidence to answer that question reliably."
    )