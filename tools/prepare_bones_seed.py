"""Recreate the fixed multi-motion train/heldout split from the published archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from tools.convert_soma_bvh_to_npz import SOURCE_BODIES, convert_clip, rig_calibration
from tools.soma_bvh import parse_bvh

REPO = Path(__file__).resolve().parents[1]
DEFAULT_DST = REPO / "src/assets/motions/reactor/bones_seed"
SPLIT = Path(__file__).with_name("bones_seed_split.json")


def prepare(dataset: Path, dst: Path, extracted: Path | None):
  manifest = json.loads(SPLIT.read_text())
  wanted = {row["source"]: (split, row["sha256"]) for split, rows in manifest.items() for row in rows}
  rig_path = dataset / "soma_shapes/soma_base_rig/soma_base_skel_minimal.bvh"
  rig = parse_bvh(rig_path)
  calibration = rig_calibration(rig_path)
  records = {s: [] for s in manifest}

  def convert(name: str, bvh: Path, stage: Path):
    split, expected_hash = wanted.pop(name)
    with bvh.open("rb") as f:
      if hashlib.file_digest(f, "sha256").hexdigest() != expected_hash:
        raise ValueError(f"Raw BVH differs from the training manifest: {name}")
    data = convert_clip(bvh, 50., rig=rig, **calibration)
    out = stage / split / (Path(name).stem + ".npz")
    np.savez(out, **data)
    length = len(data["body_pos_w"])
    records[split].append(dict(file=out.name, source=name, sha256=expected_hash,
                               frames=length, duration_s=length/50., z_nom=float(data["z_nom"][0])))

  if dst.exists() and any(dst.iterdir()):
    raise FileExistsError(f"Select an empty destination; {dst} already contains data.")
  dst.parent.mkdir(parents=True, exist_ok=True)
  with TemporaryDirectory(dir=dst.parent) as temporary:
    stage = Path(temporary) / "converted"
    for split in manifest:
      (stage / split).mkdir(parents=True)
    if extracted is not None or (dataset / "soma_uniform/bvh").is_dir():
      for name in list(wanted):
        convert(name, (extracted or dataset) / name, stage)
    else:
      with tarfile.open(dataset / "soma_uniform.tar.gz", "r|gz") as archive:
        for member in archive:
          name = member.name.removeprefix("./")
          if member.isfile() and name in wanted:
            bvh = Path(temporary) / "clip.bvh"
            with archive.extractfile(member) as f:
              bvh.write_bytes(f.read())
            convert(name, bvh, stage)
            if not wanted:
              break
    if wanted:
      raise FileNotFoundError(f"Missing archive members: {sorted(wanted)}")
    for split, rows in records.items():
      info = dict(fps=50, body_names=SOURCE_BODIES, root_body="Hips",
                  root_height=calibration["root_height"],
                  nominal_yaw_offset_deg=np.degrees(calibration["yaw"]),
                  clips=sorted(rows, key=lambda r: r["source"]))
      (stage / split / "clips.json").write_text(json.dumps(info, indent=2) + "\n")
    stage.replace(dst)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--dataset", type=Path, default=Path("/home/kiku/data/RL/ASSETS/bones-seed"))
  parser.add_argument("--dst", type=Path, default=DEFAULT_DST)
  parser.add_argument("--extracted", type=Path, help="Optional cache containing soma_uniform/bvh")
  args = parser.parse_args()
  prepare(args.dataset, args.dst, args.extracted)
