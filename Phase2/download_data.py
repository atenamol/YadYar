from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

URLS = [
    "https://archive.ics.uci.edu/static/public/349/open+university+learning+analytics+dataset.zip",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00349/OULAD.zip",
]
REQUIRED_FILES = {
    "studentInfo.csv",
    "studentVle.csv",
    "assessments.csv",
    "studentAssessment.csv",
}


def download(url: str, destination: Path) -> None:
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with destination.open("wb") as file, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc="Downloading OULAD",
        ) as progress:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)
                    progress.update(len(chunk))


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and extract OULAD.")
    parser.add_argument("--data-dir", default="data", help="Extraction directory")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "OULAD.zip"

    if REQUIRED_FILES.issubset({p.name for p in data_dir.glob("*.csv")}):
        print(f"Required files already exist in {data_dir.resolve()}")
        return

    last_error: Exception | None = None
    for url in URLS:
        try:
            print(f"Trying: {url}")
            download(url, zip_path)
            break
        except Exception as exc:  # noqa: BLE001 - clear fallback message for students
            last_error = exc
            if zip_path.exists():
                zip_path.unlink()
            print(f"Download failed from this address: {exc}")
    else:
        raise RuntimeError(
            "Could not download OULAD automatically. Download it manually from the UCI "
            "Machine Learning Repository and extract the CSV files into the data folder."
        ) from last_error

    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(data_dir)

    missing = REQUIRED_FILES - {p.name for p in data_dir.glob("*.csv")}
    if missing:
        raise FileNotFoundError(f"Missing required files after extraction: {sorted(missing)}")

    print(f"OULAD extracted successfully to {data_dir.resolve()}")


if __name__ == "__main__":
    main()
