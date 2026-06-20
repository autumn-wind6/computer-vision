# 计算机视觉期末作业：2DGS/AIGC 场景融合与 LeRobot ACT 泛化

本仓库保存期末作业的可复现实验骨架：配置、脚本、报告模板、实验记录和轻量结果。大文件不进 Git，包括 CALVIN 数据、Mip-NeRF 360 数据、第三方仓库、训练输出、视频和模型权重。

执行策略固定为 **当前 Mac 负责整理与调度，云端 NVIDIA GPU 负责 CUDA 训练与渲染**。作业截止时间为 **2026-06-23 23:59 北京时间**。

## 目标

- 题目一：真实物体多视角 2DGS、threestudio 文本到 3D、Magic123 单图到 3D、Mip-NeRF 360 背景 2DGS，并融合生成 20-40 秒漫游视频。
- 题目二：使用 LeRobot ACT，在 CALVIN `splitB` 训练基础模型，在 `splitA+B+C` 训练联合模型，在 `splitD` 做 zero-shot 测试。
- 最终提交：PDF 报告、Public GitHub 仓库、模型权重与关键资产下载链接。

## 目录结构

```text
final/
  configs/          实验配置和路径示例
  envs/             Conda 环境模板
  report/           LaTeX 报告模板
  scripts/          命令行脚本入口，按任务类型分组
    act/            ACT 训练与离线评测
    data/           数据集、采集视频、COLMAP 输入准备
    render/         融合视频渲染和帧处理
    setup/          环境检查、工作区初始化、smoke test
    utils/          指标汇总、打包、最终审计
    vision/         2DGS、AIGC 3D 生成与 PLY 融合
  src/cvfinal/      共享 Python 工具
```

外部大文件目录建议放在云端机器：

```text
/workspace/cv_final_data/
  datasets/
  captures/
  third_party/
  runs/
  exports/
  logs/
  weights/
```

## 0. 本机与云端检查

当前 Mac 可运行结构检查和报告整理：

```bash
python3 scripts/setup/check_environment.py --data-root /workspace/cv_final_data
python3 scripts/setup/run_smoke_tests.py
```

云端 GPU 机器应通过严格检查：

```bash
python scripts/setup/check_environment.py --data-root /workspace/cv_final_data --strict
```

完整一键流程的宿主机需要 `nvcc`、`nvidia-smi`、`mamba`、`git` 和 Blender。`ffmpeg` 与 COLMAP 会随 mamba 环境安装；若宿主机已安装也会优先使用。

## 1. 初始化工作区

```bash
export CVFINAL_DATA_ROOT=/workspace/cv_final_data
python scripts/setup/init_workspace.py --data-root "$CVFINAL_DATA_ROOT"
```

数据准备流程见下文各阶段命令。

### 云端一键入口

租 GPU 服务器后，建议把 `cvfinal-code.zip` 和 `cvfinal-captures.zip` 上传到服务器同一目录，解压代码后运行：

```bash
mkdir -p /root/final
unzip /root/autodl-tmp/cvfinal-code.zip -d /root/final
cp /root/autodl-tmp/cvfinal-captures.zip /root/final/
cd /root/final
export CVFINAL_DATA_ROOT=/root/autodl-tmp/cv_final_data
bash scripts/cloud_oneclick.sh all
```

AutoDL 50GB 数据盘建议开启低磁盘模式：

```bash
cd /root/final
export CVFINAL_DATA_ROOT=/root/autodl-tmp/cv_final_data
export LOW_DISK_MODE=1
export OBJECT_A_MAX_FRAMES=80
bash scripts/cloud_oneclick.sh disk-usage
bash scripts/cloud_oneclick.sh all
```

`LOW_DISK_MODE=1` 会在环境安装后清理 mamba/pip 包缓存，在准备好 CALVIN 的本地训练根目录后删除原始下载目录，并删除解压后的临时采集输入。若磁盘仍紧张，优先分阶段运行并在阶段之间执行：

```bash
bash scripts/cloud_oneclick.sh cleanup-caches
bash scripts/cloud_oneclick.sh disk-usage
```

如果第三方仓库浅克隆导致某个项目运行异常，再设置 `GIT_CLONE_DEPTH=0` 重新克隆完整仓库。

如果服务器已经有部分数据或训练中断，可按阶段继续：

```bash
bash scripts/cloud_oneclick.sh setup-envs
bash scripts/cloud_oneclick.sh clone-third-party
bash scripts/cloud_oneclick.sh prepare-inputs
bash scripts/cloud_oneclick.sh calvin
bash scripts/cloud_oneclick.sh act
bash scripts/cloud_oneclick.sh vision
bash scripts/cloud_oneclick.sh render-video
bash scripts/cloud_oneclick.sh package
bash scripts/cloud_oneclick.sh audit
```

服务器释放前必须通过最终审计：

```bash
bash scripts/cloud_oneclick.sh server-final
```

该阶段会重新收集/归一化资产 PLY、融合场景、用 Blender 渲染 24 秒漫游视频、提取报告关键帧、打包权重和关键资产，并检查是否缺文件。AutoDL 服务器建议将 `CVFINAL_DATA_ROOT` 设置在数据盘，例如 `/root/autodl-tmp/cv_final_data`。通过后下载：

```text
/root/autodl-tmp/cv_final_data/exports/cvfinal_server_artifacts.zip
/root/autodl-tmp/cv_final_data/exports/final_server_audit.json
```

`vision` 阶段需要提前准备 Mip-NeRF 360 `counter` 数据到 `$CVFINAL_DATA_ROOT/datasets/mipnerf360/counter`，或通过 `MIPNERF_COUNTER_DIR=/path/to/counter` 指向真实路径。脚本会从常见输出目录自动发现 `.ply`，并转换为融合所需的 ASCII XYZRGB PLY；若某个第三方仓库输出位置不标准，请在服务器上显式指定：

```bash
export OBJECT_A_PLY=/path/to/object_a.ply
export OBJECT_B_PLY=/path/to/object_b.ply
export OBJECT_C_PLY=/path/to/object_c.ply
bash scripts/cloud_oneclick.sh server-final
```

## 2. Mamba 环境

建议三个环境隔离：

```bash
mamba env create -f envs/environment-2dgs.yml
mamba env create -f envs/environment-aigc-3d.yml
mamba env create -f envs/environment-lerobot-act.yml
```

第三方仓库放到：

```text
$CVFINAL_DATA_ROOT/third_party/2d-gaussian-splatting
$CVFINAL_DATA_ROOT/third_party/threestudio
$CVFINAL_DATA_ROOT/third_party/magic123
$CVFINAL_DATA_ROOT/third_party/lerobot
```

不同项目依赖可能变化，以其官方 README 为准补装缺失包。

## 3. 数据下载

CALVIN LeRobot 数据使用提供的 Hugging Face 仓库：

```bash
mamba activate cvfinal-lerobot-act
python scripts/data/download_calvin_lerobot.py \
  --target "$CVFINAL_DATA_ROOT/datasets/calvin_lerobot" \
  --splits splitA splitB splitC splitD
```

固定划分：

- `splitB`：基础 ACT 模型训练。
- `splitA splitB splitC`：多环境联合 ACT 模型训练。
- `splitD`：zero-shot 测试，不参与训练。

LeRobot 训练时需要数据集根目录直接暴露 `data/` 和 `meta/`。下载完成后先准备本地训练根目录：

```bash
python scripts/data/prepare_lerobot_dataset.py \
  --dataset-root "$CVFINAL_DATA_ROOT/datasets/calvin_lerobot" \
  --splits splitB \
  --output "$CVFINAL_DATA_ROOT/datasets/calvin_prepared/act_env_b" \
  --force

python scripts/data/prepare_lerobot_dataset.py \
  --dataset-root "$CVFINAL_DATA_ROOT/datasets/calvin_lerobot" \
  --splits splitA splitB splitC \
  --output "$CVFINAL_DATA_ROOT/datasets/calvin_prepared/act_env_abc" \
  --force

python scripts/data/prepare_lerobot_dataset.py \
  --dataset-root "$CVFINAL_DATA_ROOT/datasets/calvin_lerobot" \
  --splits splitD \
  --output "$CVFINAL_DATA_ROOT/datasets/calvin_prepared/act_env_b_eval_splitD" \
  --force
```

`splitB` 和 `splitD` 默认用 symlink，`splitA+B+C` 会重编号并合并 parquet/meta，需安装 `pyarrow`。

## 4. 题目一：2DGS 与 AIGC 融合

### 4.1 背景 2DGS

默认背景为 Mip-NeRF 360 `counter`：

```bash
mamba activate cvfinal-2dgs
python scripts/vision/run_2dgs.py train \
  --repo "$CVFINAL_DATA_ROOT/third_party/2d-gaussian-splatting" \
  --source "$CVFINAL_DATA_ROOT/datasets/mipnerf360/counter" \
  --output "$CVFINAL_DATA_ROOT/runs/2dgs/background_counter" \
  --resolution 4 \
  --iterations 7000

python scripts/vision/run_2dgs.py render \
  --repo "$CVFINAL_DATA_ROOT/third_party/2d-gaussian-splatting" \
  --source "$CVFINAL_DATA_ROOT/datasets/mipnerf360/counter" \
  --output "$CVFINAL_DATA_ROOT/runs/2dgs/background_counter"
```

### 4.2 物体 A：真实多视角 2DGS

将 80-150 张环绕照片放入：

```text
$CVFINAL_DATA_ROOT/captures/object_a/images
```

如果上传的是 MP4 环绕视频，先抽帧：

```bash
python scripts/data/prepare_captures_from_video.py \
  --video /kaggle/working/captures_input/cvfinal-captures/object_video.mp4 \
  --object-a-images "$CVFINAL_DATA_ROOT/captures/object_a/images" \
  --object-c-image "$CVFINAL_DATA_ROOT/captures/object_c/foreground_candidate.png" \
  --fps 2 \
  --max-frames 150
```

`foreground_candidate.png` 只是 Magic123 的候选帧。若已上传单独的去背景图片，请复制为 `$CVFINAL_DATA_ROOT/captures/object_c/foreground.png`。

运行 COLMAP 和 2DGS：

```bash
python scripts/data/prepare_colmap_object.py \
  --images "$CVFINAL_DATA_ROOT/captures/object_a/images" \
  --workspace "$CVFINAL_DATA_ROOT/captures/object_a/colmap"

python scripts/vision/run_2dgs.py train \
  --repo "$CVFINAL_DATA_ROOT/third_party/2d-gaussian-splatting" \
  --source "$CVFINAL_DATA_ROOT/captures/object_a/colmap" \
  --output "$CVFINAL_DATA_ROOT/runs/2dgs/object_a" \
  --resolution 2 \
  --iterations 7000
```

### 4.3 物体 B：文本到 3D

```bash
mamba activate cvfinal-aigc-3d
python scripts/vision/run_threestudio_asset.py \
  --repo "$CVFINAL_DATA_ROOT/third_party/threestudio" \
  --prompt "a small colorful plastic robot figurine, clean geometry, high quality texture, studio lighting" \
  --output "$CVFINAL_DATA_ROOT/runs/aigc/object_b_text_to_3d"
```

### 4.4 物体 C：单图到 3D

先将去背景前景图保存为：

```text
$CVFINAL_DATA_ROOT/captures/object_c/foreground.png
```

再运行：

```bash
python scripts/vision/run_magic123_asset.py \
  --repo "$CVFINAL_DATA_ROOT/third_party/magic123" \
  --image "$CVFINAL_DATA_ROOT/captures/object_c/foreground.png" \
  --output "$CVFINAL_DATA_ROOT/runs/aigc/object_c_image_to_3d"
```

### 4.5 融合与视频

将背景、A、B、C 导出或采样为 ASCII PLY，至少包含 `x y z red green blue`。复制并调整 [configs/fusion_transforms.example.json](configs/fusion_transforms.example.json) 中的 `scale`、`rotation_deg`、`translation`。

```bash
python scripts/vision/merge_scene_assets.py \
  --config configs/fusion_transforms.example.json \
  --output "$CVFINAL_DATA_ROOT/exports/fusion/counter_with_assets_ascii.ply"
```

高分展示路线：在 Blender 中导入融合 PLY/mesh，设置相机轨迹，渲染 20-40 秒视频。若已有连续帧：

```bash
python scripts/render/frames_to_video.py \
  --frames "$CVFINAL_DATA_ROOT/exports/fusion/frames" \
  --pattern "%05d.png" \
  --fps 24 \
  --output "$CVFINAL_DATA_ROOT/exports/videos/counter_fusion_walkthrough.mp4"
```

## 5. 题目二：LeRobot ACT

两个配置必须保持架构和超参数一致，只改变训练 split。配置位于 [configs/act_env_b.json](configs/act_env_b.json) 和 [configs/act_env_abc.json](configs/act_env_abc.json)。

基础模型：

```bash
python scripts/act/run_act_experiment.py train \
  --config configs/act_env_b.json \
  --output "$CVFINAL_DATA_ROOT/runs/act/env_b"
```

联合模型：

```bash
python scripts/act/run_act_experiment.py train \
  --config configs/act_env_abc.json \
  --output "$CVFINAL_DATA_ROOT/runs/act/env_abc"
```

环境 D zero-shot：

```bash
python scripts/act/run_act_experiment.py eval \
  --config configs/act_env_b.json \
  --checkpoint "$CVFINAL_DATA_ROOT/runs/act/env_b/checkpoints/best" \
  --output "$CVFINAL_DATA_ROOT/runs/act/env_b_eval_d"

python scripts/act/run_act_experiment.py eval \
  --config configs/act_env_abc.json \
  --checkpoint "$CVFINAL_DATA_ROOT/runs/act/env_abc/checkpoints/best" \
  --output "$CVFINAL_DATA_ROOT/runs/act/env_abc_eval_d"
```

如果当前 LeRobot 版本的 CLI 参数发生变化，只改配置文件里的 `command_template`，不要改变 split 方案和两组超参数一致性。

## 6. 指标、报告和打包

导出 CSV 后整理表格：

```bash
python scripts/utils/collect_metrics.py \
  --inputs "$CVFINAL_DATA_ROOT/runs/act/env_b/metrics.csv" "$CVFINAL_DATA_ROOT/runs/act/env_abc/metrics.csv" \
  --output-json results/act_metrics_summary.json \
  --output-md results/act_metrics_summary.md
```

报告模板：

```bash
report/main.tex
```

打包权重与关键资产：

```bash
python scripts/utils/package_weights.py \
  --output "$CVFINAL_DATA_ROOT/weights/cvfinal_best_weights.zip" \
  --inputs "$CVFINAL_DATA_ROOT/runs/act/env_b/checkpoints/best" "$CVFINAL_DATA_ROOT/runs/act/env_abc/checkpoints/best" "$CVFINAL_DATA_ROOT/exports/fusion"
```

报告必须包含：组员姓名、学号、分工、GitHub 链接、权重网盘链接、题目一关键帧和融合视频截图、题目二训练曲线和 splitD zero-shot 表格、action chunking 分析。

## 7. Smoke Test

本仓库级轻量检查不会训练模型，只验证 JSON、PLY 融合逻辑和 wrapper dry-run：

```bash
python3 scripts/setup/run_smoke_tests.py
```
