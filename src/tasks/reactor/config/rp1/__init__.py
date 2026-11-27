from mjlab.tasks.registry import register_mjlab_task

from src.tasks.reactor.rl.runner import ReactorOnPolicyRunner

from .env_cfg import rp1_reactor_env_cfg
from .rl_cfg import rp1_reactor_runner_cfg

register_mjlab_task(
  task_id="RP1-ReActor-Flat",
  env_cfg=rp1_reactor_env_cfg(play=False),
  play_env_cfg=rp1_reactor_env_cfg(play=True),
  rl_cfg=rp1_reactor_runner_cfg(),
  runner_cls=ReactorOnPolicyRunner,
)
