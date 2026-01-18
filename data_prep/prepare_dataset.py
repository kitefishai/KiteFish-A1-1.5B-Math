import os
import json
import time
import requests
import hashlib
from pathlib import Path

from dotenv import load_dotenv

from extraction import extract_tex
from latex_cleaner import clean_latex_content
from filters import is_english, long_enough, filter_metadata_item, is_valid_tar

load_dotenv()

# ============================================================
# Use /mnt disk for all heavy data
# ============================================================
BASE_DIR    = Path(os.getenv("VOLUME"))
LATEX_DIR   = BASE_DIR / "latex_sources"
OUT_DIR     = BASE_DIR / "processed"
TMP_DIR     = BASE_DIR / "tmp"
# CHECKPOINT  = BASE_DIR / "checkpoint.txt"
MAX_RETRIES = 3

LATEX_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR.mkdir(parents=True, exist_ok=True)


async def stream_metadata(path):
    """Stream JSONL line-by-line without loading into memory."""
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


async def download_latex(arxiv_id: str):
    """Download and cache LaTeX source on /mnt."""
    dest = LATEX_DIR / f"{arxiv_id}.tar.gz"

    if dest.exists():
        return dest

    HEADERS = {"User-Agent": "slm-dataset-builder/1.0"}

    url = f"https://arxiv.org/src/{arxiv_id}"

    for attempt in range(MAX_RETRIES):
        sleep = 30
        try:
            r = requests.get(
                url,
                headers=HEADERS,
                timeout=20
            )

            if r.status_code == 200:
                dest.write_bytes(r.content)
                return dest

            if r.status_code != 200:
                if attempt == MAX_RETRIES - 1:
                    print(f"Error downloading {url}: {r.status_code}")
                # time.sleep(sleep)
                continue

        except requests.RequestException:
            # time.sleep(sleep)
            print(f"Failed to download {url}. Moving to next")

    return None


async def hash_text(text: str):
    return hashlib.md5(text.encode("utf-8")).hexdigest()


async def build_dataset(metadata_path: Path):
    print("Building dataset...", flush=True)

    stats = {
        "incorrect_id": 0,
        "blocked_by_filter": 0,
        "downloaded": 0,
        "empty_tex": 0,
        "incorrect_tar_path": 0,
        "too_short": 0,
        "non_english": 0,
        "duplicate": 0,
        "written": 0,
    }

    train_file = OUT_DIR / "train.jsonl"
    val_file   = OUT_DIR / "val.jsonl"

    train_writer = train_file.open("w", buffering=1)
    val_writer   = val_file.open("w", buffering=1)

    seen_hashes = set()
    item_count = 0
    count = 0

    # start_from = 0
    # if CHECKPOINT.exists():
    #     try:
    #         start_from = int(CHECKPOINT.read_text().strip())
    #     except ValueError:
    #         start_from = 0

    async for item in stream_metadata(metadata_path):
        item_count += 1
        # if item_count < start_from:
        #     continue

        if item_count % 1000 == 0:
            time.sleep(0.2)  # global throttle
            # CHECKPOINT.write_text(str(item_count))
            print(stats, flush=True)
            print(f"Starting item #{item_count} in 0.2 seconds....." , flush=True)


        arxiv_id = item.get("id")
        if not arxiv_id:
            stats["incorrect_id"] += 1
            continue

        # Filter BEFORE download
        if not filter_metadata_item(item):
            stats["blocked_by_filter"] += 1
            continue

        # Download LaTeX
        tar_path = await download_latex(arxiv_id)
        if tar_path is None or not is_valid_tar(tar_path):
            if tar_path and tar_path.exists():
                tar_path.unlink()
            stats["incorrect_tar_path"] += 1
            continue

        stats["downloaded"] += 1

        # Extract .tex data
        raw_tex = await extract_tex(tar_path)
        if not raw_tex:
            if tar_path and os.path.exists(tar_path):
                os.remove(tar_path)
            stats["empty_tex"] += 1
            continue

        # Clean LaTeX
        clean_txt = clean_latex_content(raw_tex)

        # Build final combined text
        final_text = f"""
Title: {item.get('title')}
Abstract: {item.get('abstract')}
Body: {clean_txt}
""".strip()

        # Quality filters
        if not long_enough(final_text):
            if tar_path and os.path.exists(tar_path):
                os.remove(tar_path)
            stats["too_short"] += 1
            continue

        if not is_english(final_text):
            if tar_path and os.path.exists(tar_path):
                os.remove(tar_path)
            stats["non_english"] += 1
            continue

        # Dedup
        h = hash_text(final_text)
        if h in seen_hashes:
            if tar_path and os.path.exists(tar_path):
                os.remove(tar_path)
            stats["duplicate"] += 1
            continue
        seen_hashes.add(h)

        # Train/Val split
        count += 1
        if count % 100 == 0:
            val_writer.write(json.dumps({"id": arxiv_id, "text": final_text}) + "\n")
        else:
            train_writer.write(json.dumps({"id": arxiv_id, "text": final_text}) + "\n")

        if count % 100 == 0:
            print(f"[INFO] Processed {count} samples", flush=True)

        stats["written"] += 1

        if tar_path and os.path.exists(tar_path):
            os.remove(tar_path)

    train_writer.close()
    val_writer.close()


if __name__ == "__main__":
    build_dataset(BASE_DIR /"arxiv-metadata-oai-snapshot.json")
