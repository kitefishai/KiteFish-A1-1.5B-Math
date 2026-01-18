import tarfile
import gzip

async def extract_tex(path, max_size=5_000_000):
    # 1️⃣ Try tar.gz
    try:
        with tarfile.open(path, "r:*") as tar:
            chunks = []
            for m in tar:
                if m.isfile() and m.name.lower().endswith(".tex"):
                    if m.size <= max_size:
                        chunks.append(
                            tar.extractfile(m)
                            .read()
                            .decode("utf-8", errors="replace")
                        )
            if chunks:
                return "\n\n".join(chunks)
    except EOFError:
        return None
    except tarfile.ReadError:
        return None

    # 2️⃣ Try gzipped single tex
    try:
        with gzip.open(path, "rb") as f:
            text = f.read(max_size).decode("utf-8", errors="replace")
            if "\\documentclass" in text or "\\begin{document}" in text:
                return text
    except:
        return None

    # 3️⃣ Try plain tex
    try:
        with open(path, "rb") as f:
            text = f.read(max_size).decode("utf-8", errors="replace")
            if "\\documentclass" in text or "\\begin{document}" in text:
                return text
    except:
        return None

    return None


async def extract_tex_from_tar(tar_path):
    """Extract text from .tar.gz in a memory-safe way."""
    chunks = []

    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            for member in tar.getmembers():
                if member.name.endswith(".tex"):
                    f = tar.extractfile(member)
                    if f:
                        try:
                            chunks.append(f.read().decode("utf-8", errors="ignore"))
                        except:
                            continue
    except tarfile.ReadError as e:
        print(f"Error extracting {tar_path}: {e}", flush=True)
        return ""
    except Exception as e:
        print(f"Error extracting {tar_path}: {e}", flush=True)
        return ""

    return "\n".join(chunks)