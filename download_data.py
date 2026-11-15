"""Download and safely extract Apache SpamAssassin's public mail corpus."""

from pathlib import Path
import tarfile
from urllib.request import urlopen


BASE_URL = "https://spamassassin.apache.org/old/publiccorpus/"
ARCHIVES = {
    "20030228_easy_ham.tar.bz2": "Apache's easy ham (non-spam) messages",
    "20030228_hard_ham.tar.bz2": "Apache's hard ham (non-spam) messages",
    "20030228_spam.tar.bz2": "Apache's spam messages",
    "20030228_spam_2.tar.bz2": "Apache's additional spam messages",
}
DATA_DIR = Path(__file__).resolve().parent / "data" / "raw"


def safe_extract(archive, destination):
    """Extract only files whose resolved path remains inside destination."""
    root = destination.resolve()
    with tarfile.open(archive, "r:bz2") as tar:
        for member in tar.getmembers():
            target = (destination / member.name).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"Unsafe path in dataset archive: {member.name}")
        tar.extractall(destination, filter="data")


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for filename, description in ARCHIVES.items():
        archive_path = DATA_DIR / filename
        print(f"Downloading {description}...")
        with urlopen(BASE_URL + filename, timeout=60) as response:
            archive_path.write_bytes(response.read())
        print(f"Extracting {filename}...")
        safe_extract(archive_path, DATA_DIR)
        archive_path.unlink()
    print(f"\nDataset is ready in: {DATA_DIR}")


if __name__ == "__main__":
    main()
