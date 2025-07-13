from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class Record:
    id: str
    audio: Path
    image: Path
    brix: float


def scan_raw(root: Path) -> List[Record]:
    records: List[Record] = []
    root = Path(root)
    for folder in root.iterdir():
        if not folder.is_dir():
            continue
        if folder.name.startswith("fold_"):
            meta_file = folder / "meta.json"
            if not meta_file.exists():
                continue
            meta = json.loads(meta_file.read_text())
            record = Record(
                id=meta.get("id", folder.name),
                audio=folder / "audio.wav",
                image=folder / "image.jpg",
                brix=float(meta.get("brix", 0)),
            )
            records.append(record)
        elif "_" in folder.name and (folder / "chu").exists():
            # legacy format dataid_label/chu/sub*/(wav,jpg)
            data_id, label = folder.name.split("_", 1)
            for sub in (folder / "chu").iterdir():
                if not sub.is_dir():
                    continue
                wavs = list(sub.glob("*.wav"))
                imgs = list(sub.glob("*.jpg"))
                if not wavs or not imgs:
                    continue
                record = Record(
                    id=f"{data_id}_{sub.name}",
                    audio=wavs[0],
                    image=imgs[0],
                    brix=float(label),
                )
                records.append(record)
    return records


def to_imf(records: List[Record], out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = {"train": [], "val": [], "test": []}
    for idx, rec in enumerate(records):
        fold = out_dir / f"fold_{idx}"
        fold.mkdir(exist_ok=True)
        (fold / "audio.wav").write_bytes(rec.audio.read_bytes())
        (fold / "image.jpg").write_bytes(rec.image.read_bytes())
        meta = {"id": rec.id, "brix": rec.brix}
        (fold / "meta.json").write_text(json.dumps(meta))
        if idx % 10 == 0:
            splits["test"].append(f"fold_{idx}")
        elif idx % 10 == 1:
            splits["val"].append(f"fold_{idx}")
        else:
            splits["train"].append(f"fold_{idx}")
    with (out_dir / "splits.yaml").open("w") as f:
        import yaml

        yaml.safe_dump(splits, f)
