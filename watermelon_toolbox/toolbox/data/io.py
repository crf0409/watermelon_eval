from pathlib import Path
from dataclasses import dataclass
from typing import List


@dataclass
class Record:
    id: str
    audio: Path
    image: Path
    brix: float


def scan_raw(root: Path) -> List[Record]:
    """Scan raw dataset directory and return records."""
    raise NotImplementedError


def to_imf(records: List[Record], out_dir: Path) -> None:
    """Convert records to intermediate format and save splits."""
    raise NotImplementedError
