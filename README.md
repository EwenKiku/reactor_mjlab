# Pi Plus ReActor

用 BONES-SEED 的 SOMA 动作训练一个共享策略。输入是人体 BVH，输出是 Pi Plus 在 MuJoCo 中的 50 Hz 运动轨迹，包含浮动基座、22 个铰链关节及 17 个对应刚体的状态。学习算法沿用仓库的 mjlab 环境、RSL-RL PPO 和上层重定向参数更新。

## 用现成权重转换新动作

在仓库根目录运行，本机已配置 Python 3.11 的 `reactor` 环境。

```bash
conda activate reactor
python -m tools.export_reactor_retarget \
  --checkpoint checkpoints/pi_plus_boneseed_base.pt \
  --src /path/to/new_motion.bvh \
  --dst outputs/new_motion
```

`--src` 也接受 BVH 目录、同一版本转换器产生的源 NPZ，以及原始 SOMA 压缩包。目录和压缩包支持 `--only`（文件名子串）、`--limit` 和 `--batch-size`。每次使用一个空输出目录。下面的例子从本机数据包取出第一条动作，可直接运行。

```bash
python -m tools.export_reactor_retarget \
  --checkpoint checkpoints/pi_plus_boneseed_base.pt \
  --src ~/data/RL/ASSETS/bones-seed/soma_uniform.tar.gz \
  --limit 1 --dst outputs/archive_example
```

默认在 `cuda:0` 上推理。CPU 回放可加 `--device cpu`，并设置 `OMP_NUM_THREADS=1`。每个输入从第 0 帧开始，初始化期间冻结参考，之后逐帧运行冻结的 actor、位置控制器和物理仿真。新动作使用自身的地面高度估计和训练得到的共享刚体偏移，额外的每片段高度修正取零。

`clips.json` 记录根部、全身刚体、根坐标系内肢体误差和残余力。整段中任一帧根部位置误差超过 1 m 或朝向误差超过 45° 时，轨迹写入 `failed/`，并记录首次越界帧。误差参照学习后的参数化目标计算。所有轨迹都连续运行到最后一帧，便于检查失败发生在什么时刻。

```bash
python -m tools.replay_npz --motion outputs/new_motion \
  --xml src/assets/robots/pi_plus/xmls/pi_plus.xml
```

NPZ 的 `joint_names` 对应 `joint_pos`（rad）和 `joint_vel`（rad/s）的列。`body_names` 对应世界系中的 `body_pos_w`（m）、`body_quat_w`（wxyz）、`body_lin_vel_w` 和 `body_ang_vel_w`。第一项 `base_link` 给出浮动基座轨迹。真机执行需要下游跟踪控制器，重定向策略本身使用了仿真根部辅助力。

## 已交付权重的仿真验证

冻结最终权重，在 CPU MuJoCo Warp 上从第 0 帧连续运行，随机种子为 42。6 条留出动作共 2,955 帧，全部通过根部终止条件。下表先对每条动作的帧取均值，最后一行对动作等权平均。

| 留出动作 | 根位置误差（cm） | 根坐标系内肢体位置误差（cm） | 刚体朝向误差（°） |
| --- | ---: | ---: | ---: |
| 侧向慢跑起步 | 19.0 | 5.3 | 23.3 |
| 转身起跑 | 6.2 | 4.8 | 19.2 |
| 深蹲 A359 | 10.7 | 7.9 | 21.2 |
| 深蹲 A360 | 9.5 | 8.0 | 20.6 |
| 向前走 | 3.5 | 4.5 | 14.6 |
| 侧向行走停止 | 16.0 | 4.9 | 23.7 |
| **平均** | **10.8** | **5.9** | **20.4** |

肢体位置误差比较各自根坐标系内的 16 个非根刚体。世界系 17 刚体平均位置误差为 12.5 cm，平均辅助力模长为 31.9 N。侧向移动的全局位移跟踪误差最大。另从原始 Uniform 和 Proportional 压缩包各读取一条训练外动作，埃及舞和转体跳均完整通过相同阈值。交付范围是仿真动作生成，尚未做 Pi Plus 实机验证。

[逐片段指标](outputs/pi_plus_base/heldout/clips.json) · [动作对照视频](outputs/pi_plus_base/preview.mp4) · [Uniform 压缩包样例](outputs/pi_plus_base/uniform_archive/clips.json) · [Proportional 压缩包样例](outputs/pi_plus_base/proportional_archive/clips.json)

## 数据和训练

已转换的数据位于 `src/assets/motions/reactor/bones_seed/{train,heldout}`。训练集包含走、转、跑、跳、蹲起过渡、坐、踢和舞蹈等 42 条动作，共 63,974 帧。留出集含 6 条完整动作，共 2,955 帧。同名动作的不同演员和镜像版本归入同一组。本次使用 SOMA Uniform 子集；对 Proportional BVH，转换器把肢段长度统一到训练骨架，并按腿长比调整平移。G1 关节数据没有参与训练。

### 原始 BVH 与训练 NPZ

原始 BVH 是动捕文件：完整 SOMA 骨架，每个关节一段局部偏移，再加上 `Zrotation Yrotation Xrotation`。只有根（Hips）带 `Xposition Yposition Zposition`。单位是厘米、Y 轴朝上，BONES-SEED 一般是 120 Hz。Uniform 和 Proportional 的差别在骨长，后者保留演员身材。

训练 NPZ 是正向运动学之后、给策略用的 17 个刚体轨迹：Hips、Chest、Head，以及左右肩、臂、前臂、手、腿、小腿、脚。每个刚体存世界系位置 `body_pos_w`、朝向 `body_quat_w`（`wxyz`），以及由相邻帧差分得到的线速度和角速度。`nominal_pos`、`nominal_quat`、`root_height`、`source_yaw` 来自固定的 Uniform 绑定姿态，同一批片段共用，不从单条表演估计。`z_nom` 是该片段双脚高度第 2 百分位的相反数，用来把脚放到地面上；`body_pos_w` 本身不预先减去它。

转换时把厘米、Y 朝上换成米、Z 朝上，再绕世界 Z 转一个固定偏航。这个角由 Uniform 骨架的绑定姿态算出，使双腿连线落到 +Y，角色在标称姿势下面对 +X。默认再重采样到 50 Hz：位置线性插值，朝向用球面插值。第 0 帧髋部的水平位置被减掉，片段从 XY 原点起步。Proportional 片段会把骨长换成训练用的 Uniform 骨架，根平移按腿长比缩放；Uniform 片段骨长不动。

NPZ 里的位置、朝向、线速度和角速度都在这同一个世界系里：Z 朝上，+X 是标定后的正前方，+Y 朝左。+X 是世界轴，不是每帧都贴着角色的脸；转身之后根朝向会离开 +X，速度仍写在世界系。每个刚体的四元数是该刚体在这个世界系中的朝向，刚体自己的局部轴仍是解剖轴。`fps`、`z_nom`、`root_height`、`source_yaw` 只是标量。

### 换机器人时的坐标对齐

BVH 到 NPZ 只处理人体。每个机器人在自己的 `calibration.yaml` 里写 17 对刚体，以及 npz 体轴到 link 轴的旋转矩阵 `R`（`v_robot = R @ v_npz`）。位置仍用标称姿势：根高之比是全局缩放 `scale`，每个刚体再得到局部位置差 `x_nom`。朝向不再现场重算，直接使用这份矩阵。之后每帧参考都是人体姿态乘 `scale`，再套上这组固定偏移。可学习的 `p_pos`、`p_ori`、`p_z` 只在这个结果上做小修正，位置和朝向各自限制在 0.5 m 和 0.5 rad。

换机器人时改该机器人目录下的 `calibration.yaml`，不改动作文件。`links` 写人体刚体到模型 body 的对应和旋转矩阵；`nominal_joints` 把机器人摆成和 SOMA 一样的双臂平举（左右 `shoulder_roll` 为 ±90°）；`foot_bodies` 和 `foot_geoms` 负责贴地。人体 NPZ 已经朝向 +X。关节目标使用模型默认顺序。

对照每个 link 的局部轴时，在 `reactor` 环境下运行：

```bash
python -m tools.visualize_link_frames
```

默认使用训练标定的 Uniform 骨架，并写出 `outputs/link_frames/soma_base_skel_minimal.npz`。左侧是原始 BVH 的全部关节，红绿蓝仍是 BVH 的局部 X/Y/Z，只把骨架转到 Z 朝上以便观看。右侧是 NPZ 的标称姿态，红绿蓝是训练用的局部轴，世界系为 Z 朝上、+X 朝前、+Y 朝左。`[` 和 `]` 切换关节，`N` 在标称姿态和动作帧之间切换。其它 BVH 用 `--bvh` 指定，输出路径用 `--out`。

### 检查 BVH 到人体 NPZ 的转换

在 `reactor` 环境下运行 `python -m tools.compare_source_motion`，即可打开 MuJoCo 原生窗口，同步播放一条训练动作。查看其他配对文件时运行

```bash
python -m tools.compare_source_motion --bvh /path/to/motion.bvh \
  --npz /path/to/train/motion.npz
```

左侧青色为原始人体骨架，右侧橙色为训练 NPZ 的 17 个刚体。两者按 NPZ 时间戳对齐，并排显示时只增加水平展示间距。空格暂停或播放，左右键逐帧，`[` 和 `]` 切换关节并显示局部坐标轴、位置差和朝向差。`O` 在并排和原位叠加之间切换。叠加时比较同名的 17 个点，Proportional 输入采用转换时的统一骨长。加 `--check-only` 可在终端查看整段误差。

选片记录及原始 BVH 的 SHA-256 保存在 [数据清单](tools/bones_seed_split.json)。若需从发布的压缩包重建这次划分，可取一个新目录运行：

```bash
python -m tools.prepare_bones_seed \
  --dataset /home/kiku/data/RL/ASSETS/bones-seed \
  --dst /tmp/pi_plus_bones_seed
```

在其他机器上先安装仓库依赖：

```bash
pip install -e ThirdParty/mjlab -e ThirdParty/rsl_rl
python -m scripts.list_envs
```

训练使用论文 Table 1 的负平方跟踪误差。逐刚体累计位置和测地朝向误差，关节动作差分只计算 20 个受控关节，同时做上层参数更新。

```bash
python -m scripts.train Pi-Plus-ReActor-Flat \
  --env.scene.num-envs 1024 --agent.save-interval 100 --agent.run-name boneseed
```

自定义训练集可通过 `--env.commands.motion.motion-files /path/to/converted/train` 指定。BVH 批量转换入口为 `python -m tools.convert_soma_bvh_to_npz --src /path/to/bvh --dst /path/to/converted/train`。默认只读训练目录。复现回归检查可运行 `python -m unittest discover -s tests -v`。

## 实现与论文的对应关系

| 位置 | 本仓库的实现 |
| --- | --- |
| `tools/soma_bvh.py`、`tools/convert_soma_bvh_to_npz.py` | 按 BVH 层级做 FK，将厘米、Y-up 转为米、Z-up。用数据集发布的固定 SOMA 姿态标定前向和侧向，逐片段重采样与估计地面高度。 |
| `config/*/calibration.yaml`、`mdp/retarget.py` | 各机器人的 17 对刚体和 npz 到 link 的旋转矩阵；位置仍由标称姿势标定，论文式 8–13 的映射。 |
| `mdp/commands.py`、`rl/runner.py` | 用物理 rollout 的有效帧计算上层损失，通过参数化参考反传并投影更新，保存和恢复共享偏移、分片段高度、标定、策略及观测归一化。 |
| `config/pi_plus/env_cfg.py`、`config/rp1/env_cfg.py`、`mdp/actions.py` | actor 接收对应刚体在机器人根系中的位置、6D 朝向、线/角速度，以及本体状态、历史动作和初始化相位。关节目标按模型默认顺序，另加 6 维辅助 wrench。 |
| `tools/export_reactor_retarget.py` | 在训练时使用的机器人模型和控制频率下执行确定性策略，导出物理状态。 |

本次训练平台为 RTX 5080、MuJoCo Warp、1,024 并行环境 × 每轮 24 步。论文实验使用 Isaac Sim、4,096 环境、20,000 轮以及其他机器人和数据集。这里沿用双层优化、残余力控制和策略条件输入；Pi Plus 的力矩尺度、初始化扰动、训练预算和数据覆盖有单独配置。论文原文见 [ReActor](doc/ReActor:%20Reinforcement%20Learning%20for%20Physics-Aware%20Motion%20Retargeting.pdf)。
