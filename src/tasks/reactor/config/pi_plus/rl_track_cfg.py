"""PPO config for the downstream Pi Plus tracking policy (paper Table 6)."""

from __future__ import annotations

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


def mlp(actor: bool = False) -> RslRlModelCfg:
  return RslRlModelCfg(
    hidden_dims=(512, 256, 128),
    activation="elu",
    obs_normalization=True,
    distribution_cfg=(
      {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"}
      if actor
      else None
    ),
  )


def pi_plus_track_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=mlp(actor=True),
    critic=mlp(),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
      normalize_advantage_per_mini_batch=False,
    ),
    obs_groups={"actor": ("actor",), "critic": ("critic",)},
    experiment_name="pi_plus_track",
    wandb_project="pi_plus_track",
    num_steps_per_env=24,
    max_iterations=50000,
    save_interval=500,
    logger="tensorboard",
  )
