import re
import tarfile
from langdetect import detect
from difflib import SequenceMatcher


async def is_valid_tar(path):
    try:
        with tarfile.open(path, "r:*"):
            return True
    except Exception:
        return False


async def is_english(text: str) -> bool:
    try:
        return detect(text) == "en"
    except:
        return False


async def long_enough(text: str, min_tokens=50) -> bool:
    return len(text.split()) >= min_tokens


async def too_similar(a: str, b: str, threshold=0.92) -> bool:
    """Fuzzy deduplication via SequenceMatcher"""
    return SequenceMatcher(None, a, b).ratio() > threshold


async def filter_metadata_item(item):

    # 1. category-based filtering
    allowed_cats = ["math", "cs", "hep-th", "hep-ph", "quant-ph", "stat.ML", "stat.TH"]

    item_cat = item.get("categories", "")
    if not any(c in item_cat for c in allowed_cats):
        return False

    # 2. abstract must be long enough
    abstract = item.get("abstract", "")
    if len(abstract.split()) < 20:
        return False

    # # 3. English-only
    # try:
    #     if detect(abstract) != "en":
    #         return False
    # except:
    #     return False

    # 4. optional: skip old papers
    # extract year from version timestamp
    if item["versions"]:
        created = item["versions"][0]["created"]

        match = re.search(r"\b(19|20)\d{2}\b", created)
        if match:
            year = int(match.group())
            if year < 1995:
                return False

    # 5. optional: skip withdrawn
    comments = item.get("comments", "") or ""
    if "withdrawn" in comments.lower():
        return False

    return True
