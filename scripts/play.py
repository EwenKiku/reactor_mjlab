"""Play a trained ReActor / tracking policy with mjlab's viewer.

Example:
  python -m scripts.play Pi-Plus-ReActor-Flat \
      --checkpoint-file logs/rsl_rl/pi_plus_reactor/<run>/model_600.pt
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

from scripts._tasks import register_tasks


def _export_play_onnx(runner, checkpoint_path: str) -> None:
  """Write an ONNX policy next to the checkpoint being played."""
  if not hasattr(runner, "export_policy_to_onnx"):
    return
  resume_path = Path(checkpoint_path)
  onnx_path = resume_path.with_suffix(".onnx")
  try:
    runner.export_policy_to_onnx(str(onnx_path.parent), onnx_path.name)
  except Exception as e:
    print(f"[WARN] ONNX export failed: {e}")
    return
  if hasattr(runner, "get_export_metadata"):
    try:
      from mjlab.rl.exporter_utils import attach_metadata_to_onnx

      metadata = runner.get_export_metadata(resume_path.parent.name)
      attach_metadata_to_onnx(str(onnx_path), metadata)
    except Exception as e:
      print(f"[WARN] ONNX metadata was not attached: {e}")
  print(f"[INFO]: Exported ONNX policy: {onnx_path}")


def _export_onnx_on_play_load() -> None:
  """Export ONNX from this play process only. Training loads are unchanged."""
  from mjlab.rl.runner import MjlabOnPolicyRunner

  original_load = MjlabOnPolicyRunner.load

  def load(self, path, load_cfg=None, strict=True, map_location=None):
    loaded = original_load(
      self, path, load_cfg=load_cfg, strict=strict, map_location=map_location
    )
    _export_play_onnx(self, path)
    return loaded

  MjlabOnPolicyRunner.load = load  # type: ignore[method-assign]


def main() -> None:
  register_tasks()
  _export_onnx_on_play_load()
  from mjlab.scripts.play import main as mjlab_main

  mjlab_main()


if __name__ == "__main__":
  main()
