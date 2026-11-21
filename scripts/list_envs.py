"""List built-in and locally registered tasks."""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

from scripts._tasks import register_tasks


def main() -> None:
  register_tasks()
  from mjlab.scripts.list_envs import main as mjlab_main

  mjlab_main()


if __name__ == "__main__":
  main()
