# ReActor（mjlab）

用 BONES-SEED 的 SOMA 动作训练共享策略，把人体动作重定向到机器人上。当前注册了两台机器人：

| 任务名 | 机器人 | 日志目录 | 模型 |
| --- | --- | --- | --- |
| `Pi-Plus-ReActor-Flat` | Pi Plus | `logs/rsl_rl/pi_plus_reactor/` | `src/assets/robots/pi_plus/xmls/pi_plus.xml` |
| `RP1-ReActor-Flat` | RP1（24 自由度） | `logs/rsl_rl/rp1_reactor/` | `src/assets/robots/rp1/mjcf/robot_24dof.xml` |

物理步长 0.005 s，控制周期每 4 步一次，策略和导出轨迹都是 50 Hz。算法是 mjlab 里的 RSL-RL PPO，加上论文的上层重定向参数更新。论文原文见 [ReActor](doc/ReActor:%20Reinforcement%20Learning%20for%20Physics-Aware%20Motion%20Retargeting.pdf)。

## 效果

下面是 RP1 导出的左右对照：左边是人体骨架，右边是仿真里的机器人。文件在 `doc/previews/`，不放在会被忽略的 `outputs/`。

<table>
<tr>
<td width="50%">走。停步转身 <code>walk_ff_stop_270_001__A072</code><br><video src="doc/previews/rp1_walk.mp4" controls muted loop playsinline width="100%"></video></td>
<td width="50%">慢跑绕圈 <code>jog_ff_loop_360_001__A072</code><br><video src="doc/previews/rp1_jog.mp4" controls muted loop playsinline width="100%"></video></td>
</tr>
<tr>
<td>前跳转身 <code>jump_ff_180_R_001__A192</code><br><video src="doc/previews/rp1_jump.mp4" controls muted loop playsinline width="100%"></video></td>
<td>舞蹈 <code>dance_retro_jackson_jump_on_R_002__A309</code><br><video src="doc/previews/rp1_dance.mp4" controls muted loop playsinline width="100%"></video></td>
</tr>
</table>

## 依赖和安装

需要 Python 3.11 和一块带 CUDA 的 NVIDIA GPU。MuJoCo Warp、PyTorch 以及其余 Python 依赖写在 `ThirdParty/mjlab/pyproject.toml` 和 `ThirdParty/rsl_rl/pyproject.toml` 里。本仓库的 `rsl_rl` 要盖过 mjlab 声明的上游版本，所以两个包都用可编辑安装，并且把本地 `rsl_rl` 放在后面。

```bash
conda create -n reactor python=3.11 -y
conda activate reactor
cd /path/to/reactor_mjlab
pip install -e ThirdParty/mjlab -e ThirdParty/rsl_rl
pip install pyyaml
python -m scripts.list_envs
```

`list_envs` 的输出里应能看到 `Pi-Plus-ReActor-Flat` 和 `RP1-ReActor-Flat`。无显示器导出视频时，导出脚本会自己把 `MUJOCO_GL` 设为 `egl`。

## 从数据到导出

一条动作经过四个落盘阶段。中间的参考姿态 `g` 不写文件，每个控制步由人体 NPZ 和标定在线算出来。

| 阶段 | 是什么 | 落在哪里 | 格式要点 |
| --- | --- | --- | --- |
| 1. 原始动捕 | SOMA BVH | BONES-SEED 压缩包或 `.bvh` 目录 | 厘米、Y 轴朝上，通常 120 Hz。完整骨架，通道顺序 `Zrotation Yrotation Xrotation`。只有根 `Hips` 带 `Xposition Yposition Zposition`。Uniform 与 Proportional 的差别是骨长，后者保留演员身材。 |
| 2. 人体训练 NPZ | 正向运动学之后的 17 个刚体，给策略当源动作 `m` | 默认 `src/assets/bones_seed_motions/train/*.npz` | 米、Z 轴朝上，50 Hz。世界系 +X 朝前、+Y 朝左。四元数 `wxyz`。 |
| 3. 机器人标定 | 人体刚体到机器人 link 的轴对齐，以及量位置偏差时的名义姿势 | `src/tasks/reactor/config/<robot>/calibration.yaml` | `R` 满足 `v_robot = R @ v_npz`。`nominal_joints` 是量这组数时的关节角。 |
| 4. 策略权重 | PPO actor/critic，以及学到的重定向参数 | `logs/rsl_rl/<experiment>/<时间戳>/model_*.pt` | 版本 2。含 `retarget/p_pos`、`p_ori`、`p_z`，以及标定时的 `R_nom`、`x_nom`、`scale`。加载时用检查点里的标定盖过 yaml。 |
| 5. 机器人轨迹 | 冻结策略在物理仿真里走出的轨迹 | `outputs/<robot>/*.npz` | 50 Hz。关节位置/速度，以及标定刚体的世界系位姿和速度。 |

选片清单在 `tools/bones_seed_split.json`：`train` 与 `heldout` 两组，每行是压缩包内的 BVH 路径和原始文件的 SHA-256。训练读的是转换后的 NPZ，不读这份 JSON。

### 人体训练 NPZ 的字段

17 个刚体，顺序固定为：Hips、Chest、Head，左肩、左上臂、左前臂、左手，右侧对称的四个，左腿、左小腿、左脚，右侧对称的三个。

| 字段 | 形状 | 含义 |
| --- | --- | --- |
| `fps` | 标量 | 50 |
| `body_names` | 17 | 上面的名字 |
| `body_pos_w` | `(T, 17, 3)` | 世界系位置，米 |
| `body_quat_w` | `(T, 17, 4)` | 世界系朝向，`wxyz` |
| `body_lin_vel_w` | `(T, 17, 3)` | 由相邻帧位置差分 |
| `body_ang_vel_w` | `(T, 17, 3)` | 由相邻帧朝向差分 |
| `nominal_pos`、`nominal_quat` | `(17, 3)`、`(17, 4)` | Uniform 绑定姿态，同一批片段共用 |
| `root_height` | 标量 | 绑定姿态的根高度，米 |
| `source_yaw` | 标量 | 把角色转到面对 +X 的固定偏航，弧度 |
| `z_nom` | 标量 | 该片段双脚高度第 2 百分位的相反数。脚贴地用它，`body_pos_w` 本身不减 |
| `quaternion_order` | 字符串 | `wxyz` |

转换时把厘米、Y 朝上换成米、Z 朝上，再绕世界 Z 转 `source_yaw`。这个角来自 Uniform 骨架的绑定姿态，使双腿连线落到 +Y，角色面对 +X。位置线性插值、朝向球面插值，重采样到 50 Hz。第 0 帧髋部的水平位置被减掉。Proportional 片段会把骨长换成 Uniform 骨架，根平移按腿长比缩放。

NPZ 里的位置和速度都在这个世界系里。+X 是标定后的世界前向，不是每帧贴着角色的脸；转身后根朝向会离开 +X。每个刚体的局部轴仍是变换后的 SOMA 关节轴。

同目录的 `clips.json` 只是索引：帧率、刚体名、每条片段的帧数、`z_nom` 和来源路径。

### 标定和在线参考

`calibration.yaml` 不改人体文件。`links` 的每一项是人体刚体、机器人 body，以及旋转矩阵 `R`。位置偏差不写在 yaml 里：环境启动时按 `nominal_joints` 摆好机器人，用根高之比得到全局缩放 `scale`，再得到每个刚体在人体坐标系里的常数 `x_nom`。朝向使用 yaml 里的 `R`，存进检查点时的 `R_nom` 是它的转置，于是 `R_robot = R_source @ R_nom`。

之后每一帧的参考 `g` 是人体姿态乘 `scale`，再套上这组固定偏移。可学习的 `p_pos`、`p_ori`、`p_z` 只做小修正，位置和朝向各自限制在 0.5 m 和 0.5 rad。`p_z` 每个片段一个。`nominal_joints` 必须和写 `R` 时的姿势是同一次摆放；只改关节角、不重算 `R`，朝向标定会和位置标定错开。

RP1 没有单独的头 link，`Head` 和 `Chest` 都对应 `waist_yaw_link`。关节目标使用模型里的默认铰链顺序，不另写关节表。

### 导出的机器人 NPZ

只保存通过根部阈值的片段。任一帧根位置误差超过 1 m、根朝向误差超过 45°，或仿真出现非有限值时，不写 NPZ，并删掉输出目录里的同名文件。失败原因留在 `clips.json`。

| 字段 | 含义 |
| --- | --- |
| `fps` | 50 |
| `joint_names` | 与 `joint_pos`（rad）、`joint_vel`（rad/s）的列对齐 |
| `body_names` | 与下面四个 `body_*` 的刚体维对齐，第一项是根，例如 `base_link` |
| `quaternion_order` | `wxyz` |
| `anchor_body` | 根 body 名 |
| `body_pos_w`、`body_quat_w` | 世界系位姿。环境原点已减去 |
| `body_lin_vel_w`、`body_ang_vel_w` | 世界系速度 |

真机执行还要下游跟踪控制器。重定向策略在仿真里使用了根部辅助力。

## 准备训练集

发布包里已有 `soma_uniform.tar.gz` 时，按清单重建 train / heldout。目标目录必须是空的。训练默认读取 `src/assets/bones_seed_motions/train`，所以把输出直接写到这个目录：

```bash
python -m tools.prepare_bones_seed \
  --dataset /path/to/bones-seed \
  --dst src/assets/bones_seed_motions
```

已经解压出 `soma_uniform/bvh` 时加上 `--extracted /path/to/that/tree`。脚本会核对每条 BVH 的 SHA-256，再做成 50 Hz 的人体 NPZ。

手里是零散 BVH、不走这份清单时：

```bash
python -m tools.convert_soma_bvh_to_npz \
  --src /path/to/bvh_or_dir \
  --dst /path/to/converted/train
```

`--fps` 默认 50，`--limit N` 只转前 N 条，`--rig` 可换绑定骨架。这样得到的目录用下面的 `--env.commands.motion.motion-files` 指给训练。

转换完可以看人体骨架是否和 NPZ 重合：

```bash
python -m tools.compare_source_motion --check-only
python -m tools.visualize_link_frames
```

## 训练

在仓库根目录、`reactor` 环境里运行。默认读 `src/assets/bones_seed_motions/train`，1024 个并行环境只是示例，显存不够就减小。

```bash
python -m scripts.train Pi-Plus-ReActor-Flat \
  --env.scene.num-envs 1024 \
  --agent.run-name boneseed

python -m scripts.train RP1-ReActor-Flat \
  --env.scene.num-envs 1024 \
  --agent.run-name boneseed
```

自备的人体 NPZ 目录：

```bash
python -m scripts.train RP1-ReActor-Flat \
  --env.commands.motion.motion-files /path/to/converted/train \
  --env.scene.num-envs 1024
```

每次迭代每个环境采集 24 步。权重默认每 200 次迭代存一次，最长 20000 次迭代，写到 `logs/rsl_rl/<experiment_name>/<时间戳>/`。TensorBoard 日志在同一目录。

看一条训练中的策略：

```bash
python -m scripts.play RP1-ReActor-Flat \
  --checkpoint-file logs/rsl_rl/rp1_reactor/<时间戳>/model_600.pt
```

`play` 会在检查点旁边试着导出同名 `.onnx`。回归测试：

```bash
python -m unittest discover -s tests -v
```

## 导出轨迹

导出使用检查点里保存的标定和 `p_z`，不会改用后来手改的 yaml。单条输入会写对照视频；目录默认只写 NPZ。仿真只建一次，默认 32 条并行。输出目录非空时必须加 `--resume`。

```bash
python -m tools.export_reactor_retarget \
  --robot rp1 \
  --checkpoint logs/rsl_rl/rp1_reactor/<时间戳>/model_36000.pt \
  --src src/assets/bones_seed_motions/train \
  --dst outputs/rp1 \
  --resume
```

`--src` 可以是一条 BVH、一个 BVH 或 NPZ 目录，或 SOMA 的 `tar.gz`。其它常用参数：

| 参数 | 作用 |
| --- | --- |
| `--robot {pi_plus,rp1}` | 不写时按检查点里的刚体名单自动选 |
| `--only 子串` | 只导出文件名里带这段文字的片段 |
| `--limit N` | 只取前 N 条 |
| `--batch-size 32` | 并行环境数。显存够可以加大 |
| `--video` | 目录导出时也写左右对照视频 |
| `--no-video` | 连单条导出也不写视频 |
| `--device cuda:0` | CPU 回放用 `--device cpu`，并设 `OMP_NUM_THREADS=1` |
| `--resume` | 跳过已经写出或已经记为失败的片段 |

运动学回放导出的机器人 NPZ，不跑物理：

```bash
python -m tools.replay_npz \
  --motion outputs/rp1 \
  --xml src/assets/robots/rp1/mjcf/robot_24dof.xml
```

Pi Plus 把 `--xml` 换成 `src/assets/robots/pi_plus/xmls/pi_plus.xml`。空格暂停，左右方向键切片段。

## 脚本

| 路径 | 作用 |
| --- | --- |
| `tools/soma_bvh.py` | BVH 解析、正向运动学，以及厘米 Y-up 到米 Z-up 的坐标变换。被别的脚本调用，没有自己的命令行入口。 |
| `tools/convert_soma_bvh_to_npz.py` | 把 BVH 文件或目录转成人体训练 NPZ，并写 `clips.json`。 |
| `tools/prepare_bones_seed.py` | 按 `tools/bones_seed_split.json` 从发布包抽出 train / heldout，核对 SHA-256 后转换。 |
| `tools/bones_seed_split.json` | 选片清单，不是可执行脚本。 |
| `tools/compare_source_motion.py` | 并排播放原始 BVH 和人体 NPZ。`--check-only` 只在终端打印误差。 |
| `tools/visualize_link_frames.py` | 左边画原始 BVH 关节轴，右边画训练 NPZ 的刚体轴。 |
| `scripts/list_envs.py` | 列出已注册任务。 |
| `scripts/train.py` | 训练。任务名后的 `--env.*`、`--agent.*` 覆盖配置。 |
| `scripts/play.py` | 用查看器播放检查点，并尝试导出 ONNX。 |
| `tools/export_reactor_retarget.py` | 冻结 actor，在对应机器人的物理频率下滚出轨迹 NPZ。 |
| `tools/replay_npz.py` | 把导出的关节轨迹写进 `qpos`，在 MuJoCo 窗口里运动学播放。 |

## 和论文的对应关系

| 论文里的步骤 | 本仓库 |
| --- | --- |
| 源动作 `m` | `tools/soma_bvh.py`、`tools/convert_soma_bvh_to_npz.py`：BVH 层级 FK，厘米 Y-up 到米 Z-up，固定 Uniform 姿态标定前向和侧向，50 Hz 重采样，每段估计 `z_nom`。 |
| 名义标定，式 8–9 | `config/*/calibration.yaml` 的 `R` 与 `nominal_joints`。位置在启动时算出 `scale` 和 `x_nom`，朝向用 `R`。 |
| 参数化映射 `m → g`，式 10–13 | `src/tasks/reactor/mdp/retarget.py`。`g` 是策略看到的参考，不另存文件。 |
| 上层损失和参数更新 | `mdp/commands.py` 收集有效帧，`rl/runner.py` 对 `p` 做一步更新，并把 `p`、标定和策略一起写进检查点。 |
| 下层 PPO | `config/pi_plus/rl_cfg.py`、`config/rp1/rl_cfg.py`。actor 看到根系中的参考刚体、相位、本体速度和关节、上一拍动作。动作是模型默认顺序的关节目标，再加 6 维根部辅助力，见 `mdp/actions.py`。 |
| 物理状态 `s` | 导出脚本在训练用的模型和 50 Hz 下执行确定性策略，写成机器人 NPZ。 |

跟踪奖励是论文 Table 1 的负平方误差，对刚体位置和测地朝向累加。Pi Plus 的已交付权重在 CPU 上、种子 42、从第 0 帧连续跑过 6 条留出动作共 2,955 帧，全部通过根部阈值。下表先对每条动作的帧取均值，最后一行对动作等权平均。

| 留出动作 | 根位置误差（cm） | 根坐标系内肢体位置误差（cm） | 刚体朝向误差（°） |
| --- | ---: | ---: | ---: |
| 侧向慢跑起步 | 19.0 | 5.3 | 23.3 |
| 转身起跑 | 6.2 | 4.8 | 19.2 |
| 深蹲 A359 | 10.7 | 7.9 | 21.2 |
| 深蹲 A360 | 9.5 | 8.0 | 20.6 |
| 向前走 | 3.5 | 4.5 | 14.6 |
| 侧向行走停止 | 16.0 | 4.9 | 23.7 |
| **平均** | **10.8** | **5.9** | **20.4** |

肢体位置误差比较各自根坐标系内的 16 个非根刚体。世界系 17 刚体平均位置误差为 12.5 cm，平均辅助力模长为 31.9 N。这是 Pi Plus 仿真结果，不是 RP1，也还没有实机跟踪。RP1 的对照片见上面的效果一节。
