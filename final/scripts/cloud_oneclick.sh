#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
if [ -n "${CVFINAL_DATA_ROOT:-}" ]; then
  DATA_ROOT="$CVFINAL_DATA_ROOT"
elif [ -d /root/autodl-tmp ]; then
  DATA_ROOT="/root/autodl-tmp/cv_final_data"
else
  DATA_ROOT="/workspace/cv_final_data"
fi
export CVFINAL_DATA_ROOT="$DATA_ROOT"
THIRD_PARTY_ROOT="${THIRD_PARTY_ROOT:-$DATA_ROOT/third_party}"
CAPTURES_ZIP="${CAPTURES_ZIP:-$PROJECT_ROOT/cvfinal-captures.zip}"
CAPTURES_INPUT_ROOT="${CAPTURES_INPUT_ROOT:-$DATA_ROOT/captures_input}"
MIPNERF_COUNTER_DIR="${MIPNERF_COUNTER_DIR:-$DATA_ROOT/datasets/mipnerf360/counter}"
PROMPT_OBJECT_B="${PROMPT_OBJECT_B:-a small colorful plastic robot figurine, clean geometry, high quality texture, studio lighting}"
FUSION_CONFIG="${FUSION_CONFIG:-$PROJECT_ROOT/configs/fusion_transforms.example.json}"
FUSION_PLY="${FUSION_PLY:-$DATA_ROOT/exports/fusion/counter_with_assets_ascii.ply}"
FUSION_VIDEO="${FUSION_VIDEO:-$DATA_ROOT/exports/videos/counter_fusion_walkthrough.mp4}"
REPORT_ASSETS_DIR="${REPORT_ASSETS_DIR:-$DATA_ROOT/exports/report_assets}"
SERVER_BUNDLE="${SERVER_BUNDLE:-$DATA_ROOT/exports/cvfinal_server_artifacts.zip}"

ENV_2DGS="${ENV_2DGS:-cvfinal-2dgs}"
ENV_AIGC="${ENV_AIGC:-cvfinal-aigc-3d}"
ENV_ACT="${ENV_ACT:-cvfinal-lerobot-act}"
MAMBA_EXE="${MAMBA_EXE:-mamba}"
BLENDER_EXE="${BLENDER_EXE:-blender}"
FFMPEG_EXE="${FFMPEG_EXE:-ffmpeg}"
LOW_DISK_MODE="${LOW_DISK_MODE:-0}"
OBJECT_A_FPS="${OBJECT_A_FPS:-2}"
OBJECT_A_MAX_FRAMES="${OBJECT_A_MAX_FRAMES:-150}"
GIT_CLONE_DEPTH="${GIT_CLONE_DEPTH:-1}"
# Fast-track defaults for a first complete pass on a single 2080 Ti.
ACT_ABC_STEPS="${ACT_ABC_STEPS:-25000}"
ACT_BATCH_SIZE="${ACT_BATCH_SIZE:-12}"
ACT_NUM_WORKERS="${ACT_NUM_WORKERS:-6}"
ACT_EVAL_EPISODES="${ACT_EVAL_EPISODES:-50}"
D2GS_ITERATIONS="${D2GS_ITERATIONS:-4000}"
D2GS_BG_RESOLUTION="${D2GS_BG_RESOLUTION:-6}"
D2GS_OBJECT_RESOLUTION="${D2GS_OBJECT_RESOLUTION:-4}"
THREESTUDIO_MAX_STEPS="${THREESTUDIO_MAX_STEPS:-4000}"
MAGIC123_ITERS="${MAGIC123_ITERS:-3000}"
FUSION_VIDEO_DURATION="${FUSION_VIDEO_DURATION:-12}"
FUSION_VIDEO_SAMPLES="${FUSION_VIDEO_SAMPLES:-32}"
FUSION_VIDEO_WIDTH="${FUSION_VIDEO_WIDTH:-960}"
FUSION_VIDEO_HEIGHT="${FUSION_VIDEO_HEIGHT:-540}"
if [ -z "${PYTHON_EXE:-}" ]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_EXE="python"
  else
    PYTHON_EXE="python3"
  fi
fi

export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$DATA_ROOT/mamba}"
export CONDA_ENVS_PATH="${CONDA_ENVS_PATH:-$MAMBA_ROOT_PREFIX/envs}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$MAMBA_ROOT_PREFIX/pkgs}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$DATA_ROOT/cache/pip}"
export HF_HOME="${HF_HOME:-$DATA_ROOT/cache/huggingface}"
if [ -z "${HF_ENDPOINT:-}" ] && [ -d /root/autodl-tmp ]; then
  export HF_ENDPOINT="https://hf-mirror.com"
fi
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TORCH_HOME="${TORCH_HOME:-$DATA_ROOT/cache/torch}"
export WANDB_DIR="${WANDB_DIR:-$DATA_ROOT/logs/wandb}"

REPO_2DGS="${REPO_2DGS:-$THIRD_PARTY_ROOT/2d-gaussian-splatting}"
REPO_THREESTUDIO="${REPO_THREESTUDIO:-$THIRD_PARTY_ROOT/threestudio}"
REPO_MAGIC123="${REPO_MAGIC123:-$THIRD_PARTY_ROOT/magic123}"
REPO_LEROBOT="${REPO_LEROBOT:-$THIRD_PARTY_ROOT/lerobot}"

URL_2DGS="${URL_2DGS:-https://github.com/hbb1/2d-gaussian-splatting.git}"
URL_THREESTUDIO="${URL_THREESTUDIO:-https://github.com/threestudio-project/threestudio.git}"
URL_MAGIC123="${URL_MAGIC123:-https://github.com/guochengqian/Magic123.git}"
URL_LEROBOT="${URL_LEROBOT:-https://github.com/huggingface/lerobot.git}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/cloud_oneclick.sh <stage>

Stages:
  all                 setup envs, run all experiments, render video, package, and audit
  init                create external data directories and run a strict CUDA/tool check
  setup-envs          create/update the three mamba environments
  clone-third-party   clone 2DGS, threestudio, Magic123, and LeRobot if missing
  prepare-inputs      unzip cvfinal-captures.zip and extract object-A frames/object-C foreground
  calvin              download CALVIN LeRobot splits and prepare local split roots
  act                 train splitB and splitA+B+C ACT models, then evaluate both on splitD
  finish              resume/skip finished ACT steps, then vision + render + package + audit
  download-mipnerf    download and extract the Mip-NeRF 360 counter scene if missing
  wait-and-finish     download counter, wait for ACT jobs to finish, then run finish
  vision              run 2DGS/AIGC wrappers, normalize PLY assets, and merge the scene
  prepare-fusion      normalize discovered/provided asset PLY files for fusion
  render-video        render the fused scene video and extract keyframes
  package             collect metrics and zip checkpoints/assets
  audit               fail if any required final server artifact is missing
  server-final        prepare fusion assets, merge, render, package, and audit existing runs
  disk-usage          print disk usage for the project/data roots
  cleanup-caches      remove package/download caches and temporary capture input
  smoke               run local smoke tests only

Important environment variables:
  CVFINAL_DATA_ROOT     default: /root/autodl-tmp/cv_final_data on AutoDL, otherwise /workspace/cv_final_data
  CAPTURES_ZIP          default: ./cvfinal-captures.zip
  MIPNERF_COUNTER_DIR   default: $CVFINAL_DATA_ROOT/datasets/mipnerf360/counter
  PROMPT_OBJECT_B       text prompt for threestudio object B
  OBJECT_A_PLY          optional explicit PLY source for object A
  OBJECT_B_PLY          optional explicit PLY source for object B
  OBJECT_C_PLY          optional explicit PLY source for object C
  BLENDER_EXE           default: blender
  FFMPEG_EXE            default: ffmpeg; falls back to the cvfinal-2dgs mamba env if missing
  LOW_DISK_MODE         set to 1 on small disks; copies prepared CALVIN split roots, then removes the raw download
  OBJECT_A_MAX_FRAMES   default: 150; lower this to save capture/colmap space
  GIT_CLONE_DEPTH       default: 1 for shallow third-party clones; set to 0 for full history
  FORCE_RETRAIN         set to 1 to rerun ACT even if best checkpoints already exist
  FORCE_REEVAL          set to 1 to rerun splitD eval even if output dirs already exist
  D2GS_ITERATIONS       default: 4000 fast pass; increase after the pipeline works
  THREESTUDIO_MAX_STEPS default: 4000 fast pass for object B
  MAGIC123_ITERS        default: 3000 fast pass for object C coarse stage

Example:
  export CVFINAL_DATA_ROOT=/root/autodl-tmp/cv_final_data
  bash scripts/cloud_oneclick.sh all
EOF
}

log() {
  printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"
}

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing command: $1" >&2
    return 127
  fi
}

is_low_disk_mode() {
  [ "$LOW_DISK_MODE" = "1" ] || [ "$LOW_DISK_MODE" = "true" ] || [ "$LOW_DISK_MODE" = "yes" ]
}

act_best_checkpoint() {
  local output="$1"
  [ -d "$output/checkpoints/best" ] && [ -n "$(ls -A "$output/checkpoints/best" 2>/dev/null)" ]
}

act_eval_done() {
  local output="$1"
  [ -d "$output" ] && [ -n "$(ls -A "$output" 2>/dev/null)" ]
}

mipnerf_counter_ready() {
  local dir="$1"
  [ -d "$dir" ] && { [ -d "$dir/images" ] || [ -d "$dir/sparse" ] || [ -f "$dir/transforms_train.json" ]; }
}

stage_download_mipnerf() {
  if mipnerf_counter_ready "$MIPNERF_COUNTER_DIR"; then
    log "Mip-NeRF 360 counter already present at $MIPNERF_COUNTER_DIR"
    return 0
  fi

  need_cmd wget
  need_cmd unzip

  local url="${MIPNERF360_URL:-http://storage.googleapis.com/gresearch/refraw360/360_v2.zip}"
  local zip_dir="$DATA_ROOT/cache/downloads"
  local zip_path="$zip_dir/360_v2.zip"
  local tmp_extract="$DATA_ROOT/cache/mipnerf360_extract"

  mkdir -p "$zip_dir" "$(dirname "$MIPNERF_COUNTER_DIR")"
  local expected_bytes="${MIPNERF360_ZIP_BYTES:-12535427936}"
  log "Downloading/resuming Mip-NeRF 360 dataset (~12GB) to $zip_path"
  wget -c -O "$zip_path" "$url"

  if [ ! -f "$zip_path" ] || [ "$(stat -c%s "$zip_path")" -lt "$expected_bytes" ]; then
    echo "Mip-NeRF download incomplete at $zip_path" >&2
    return 2
  fi
  if ! unzip -tq "$zip_path" "counter/*" >/dev/null 2>&1; then
    echo "Mip-NeRF archive failed integrity check: $zip_path" >&2
    return 2
  fi

  log "Extracting counter scene only"
  rm -rf "$tmp_extract"
  mkdir -p "$tmp_extract"
  unzip -q "$zip_path" "counter/*" -d "$tmp_extract"
  rm -rf "$MIPNERF_COUNTER_DIR"
  mv "$tmp_extract/counter" "$MIPNERF_COUNTER_DIR"
  rm -rf "$tmp_extract"

  if [ "${MIPNERF_DELETE_ZIP:-1}" = "1" ]; then
    rm -f "$zip_path"
  fi

  if ! mipnerf_counter_ready "$MIPNERF_COUNTER_DIR"; then
    echo "Failed to prepare Mip-NeRF counter scene at $MIPNERF_COUNTER_DIR" >&2
    return 2
  fi
  log "Mip-NeRF counter ready at $MIPNERF_COUNTER_DIR"
}

stage_wait_and_finish() {
  stage_download_mipnerf

  log "Waiting for ACT training/eval jobs to finish before running finish"
  while pgrep -f "lerobot-train|run_act_experiment.py (train|eval)" >/dev/null 2>&1; do
    sleep 60
  done

  local abc_eval_dir="$DATA_ROOT/runs/act/env_abc_eval_d"
  local wait_loops=0
  while ! act_eval_done "$abc_eval_dir"; do
    if pgrep -f "lerobot-train|run_act_experiment.py (train|eval)" >/dev/null 2>&1; then
      sleep 60
      continue
    fi
    wait_loops=$((wait_loops + 1))
    if [ "$wait_loops" -ge 120 ]; then
      echo "Timed out waiting for ACT eval outputs at $abc_eval_dir" >&2
      return 2
    fi
    sleep 30
  done

  stage_finish
}

train_act_if_needed() {
  local config="$1"
  local output="$2"
  local label="$3"
  if act_best_checkpoint "$output" && [ "${FORCE_RETRAIN:-0}" != "1" ]; then
    log "Skipping $label ACT training; best checkpoint already exists at $output/checkpoints/best"
    return 0
  fi
  log "Training ACT model: $label"
  mamba_exec "$ENV_ACT" python "$PROJECT_ROOT/scripts/act/run_act_experiment.py" train \
    --repo "$REPO_LEROBOT" \
    --config "$config" \
    --output "$output"
}

eval_act_if_needed() {
  local config="$1"
  local output="$2"
  local checkpoint="$3"
  local label="$4"
  if act_eval_done "$output" && [ "${FORCE_REEVAL:-0}" != "1" ]; then
    log "Skipping $label ACT eval; output already exists at $output"
    return 0
  fi
  log "Evaluating ACT model: $label"
  mamba_exec "$ENV_ACT" python "$PROJECT_ROOT/scripts/act/run_act_experiment.py" eval \
    --repo "$REPO_LEROBOT" \
    --config "$config" \
    --checkpoint "$checkpoint" \
    --output "$output"
}

mamba_env_exists() {
  "$MAMBA_EXE" env list | awk '{print $1}' | grep -Fxq "$1"
}

mamba_exec() {
  local env_name="$1"
  shift
  "$MAMBA_EXE" run -n "$env_name" "$@"
}

run_ffmpeg() {
  if command -v "$FFMPEG_EXE" >/dev/null 2>&1 && "$FFMPEG_EXE" -version >/dev/null 2>&1; then
    "$FFMPEG_EXE" "$@"
  elif [ -x "$MAMBA_ROOT_PREFIX/envs/$ENV_2DGS/bin/ffmpeg" ]; then
    "$MAMBA_ROOT_PREFIX/envs/$ENV_2DGS/bin/ffmpeg" "$@"
  else
    mamba_exec "$ENV_2DGS" ffmpeg "$@"
  fi
}

print_disk_usage() {
  log "Filesystem usage"
  df -h / || true
  if [ -d /root/autodl-tmp ]; then
    df -h /root/autodl-tmp || true
  fi

  log "Project/data usage"
  for path in \
    "$PROJECT_ROOT" \
    "$DATA_ROOT" \
    "$MAMBA_ROOT_PREFIX" \
    "$CONDA_ENVS_PATH" \
    "$CONDA_PKGS_DIRS" \
    "$PIP_CACHE_DIR" \
    "$HF_HOME" \
    "$THIRD_PARTY_ROOT" \
    "$DATA_ROOT/datasets" \
    "$DATA_ROOT/captures" \
    "$DATA_ROOT/runs" \
    "$DATA_ROOT/exports" \
    "$DATA_ROOT/weights"; do
    if [ -e "$path" ]; then
      du -sh "$path" 2>/dev/null || true
    fi
  done
}

ensure_env() {
  local env_name="$1"
  local env_file="$2"
  if mamba_env_exists "$env_name"; then
    log "Updating mamba env $env_name"
    "$MAMBA_EXE" env update -y -n "$env_name" -f "$env_file" --prune
  else
    log "Creating mamba env $env_name"
    "$MAMBA_EXE" env create -y -f "$env_file"
  fi
}

clone_if_missing() {
  local url="$1"
  local target="$2"
  if [ -d "$target/.git" ]; then
    log "Third-party repo exists: $target"
  else
    log "Cloning $url -> $target"
    mkdir -p "$(dirname "$target")"
    if [ "$GIT_CLONE_DEPTH" = "0" ]; then
      git clone --recursive "$url" "$target"
    else
      git clone --depth "$GIT_CLONE_DEPTH" --recursive --shallow-submodules "$url" "$target"
    fi
  fi
}

install_2dgs_python_deps() {
  mamba_exec "$ENV_2DGS" python -m pip install matplotlib mediapy open3d trimesh scikit-image
}

install_2dgs_cuda_extensions() {
  local submodules_dir="$REPO_2DGS/submodules"
  if [ ! -d "$submodules_dir/diff-surfel-rasterization" ]; then
    echo "Missing 2DGS submodules under $submodules_dir" >&2
    echo "Reclone with: git clone --recursive $URL_2DGS $REPO_2DGS" >&2
    return 2
  fi
  if mamba_exec "$ENV_2DGS" python -c "import torch; import diff_surfel_rasterization; from simple_knn._C import distCUDA2" >/dev/null 2>&1; then
    log "2DGS CUDA extensions already installed in $ENV_2DGS"
    return 0
  fi
  log "Installing 2DGS CUDA extensions into $ENV_2DGS"
  touch "$submodules_dir/simple-knn/simple_knn/__init__.py"
  for pkg in diff-surfel-rasterization simple-knn; do
    mamba_exec "$ENV_2DGS" python -m pip install -e "$submodules_dir/$pkg" --no-build-isolation
  done
  mamba_exec "$ENV_2DGS" python -c "import torch; import diff_surfel_rasterization; from simple_knn._C import distCUDA2"
}

ensure_2dgs_ready() {
  if ! mamba_exec "$ENV_2DGS" python -c "import mediapy, open3d, trimesh, skimage, matplotlib" >/dev/null 2>&1; then
    log "Installing 2DGS python dependencies into $ENV_2DGS"
    install_2dgs_python_deps
  fi
  install_2dgs_cuda_extensions
}

install_editable_if_possible() {
  local env_name="$1"
  local repo="$2"
  if [ -f "$repo/pyproject.toml" ] || [ -f "$repo/setup.py" ]; then
    log "Installing editable repo into $env_name: $repo"
    mamba_exec "$env_name" python -m pip install -e "$repo"
  else
    log "No pyproject.toml/setup.py in $repo; skipping editable install"
  fi
}

stage_init() {
  log "Project root: $PROJECT_ROOT"
  log "Data root: $DATA_ROOT"
  need_cmd "$PYTHON_EXE"
  mkdir -p "$DATA_ROOT" "$CONDA_ENVS_PATH" "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$HF_HOME" "$TORCH_HOME" "$WANDB_DIR"
  "$PYTHON_EXE" "$PROJECT_ROOT/scripts/setup/init_workspace.py" --data-root "$DATA_ROOT"
  "$PYTHON_EXE" "$PROJECT_ROOT/scripts/setup/check_environment.py" --data-root "$DATA_ROOT" --strict
}

stage_setup_envs() {
  need_cmd "$MAMBA_EXE"
  mkdir -p "$CONDA_ENVS_PATH" "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR"
  ensure_env "$ENV_2DGS" "$PROJECT_ROOT/envs/environment-2dgs.yml"
  ensure_env "$ENV_AIGC" "$PROJECT_ROOT/envs/environment-aigc-3d.yml"
  ensure_env "$ENV_ACT" "$PROJECT_ROOT/envs/environment-lerobot-act.yml"
  if is_low_disk_mode; then
    stage_cleanup_caches
  fi
}

stage_clone_third_party() {
  need_cmd git
  clone_if_missing "$URL_2DGS" "$REPO_2DGS"
  clone_if_missing "$URL_THREESTUDIO" "$REPO_THREESTUDIO"
  clone_if_missing "$URL_MAGIC123" "$REPO_MAGIC123"
  clone_if_missing "$URL_LEROBOT" "$REPO_LEROBOT"
  install_editable_if_possible "$ENV_ACT" "$REPO_LEROBOT"
}

stage_prepare_inputs() {
  mkdir -p "$CAPTURES_INPUT_ROOT"
  if [ -f "$CAPTURES_ZIP" ]; then
    need_cmd unzip
    log "Unzipping captures: $CAPTURES_ZIP"
    unzip -o -q "$CAPTURES_ZIP" -d "$CAPTURES_INPUT_ROOT"
  else
    log "Capture zip not found, skipping unzip: $CAPTURES_ZIP"
  fi

  local video="$CAPTURES_INPUT_ROOT/cvfinal-captures/object_video.mp4"
  local foreground="$CAPTURES_INPUT_ROOT/cvfinal-captures/object_c_foreground.png"

  if [ -f "$video" ]; then
    log "Extracting object-A frames from $video"
    mamba_exec "$ENV_2DGS" python "$PROJECT_ROOT/scripts/data/prepare_captures_from_video.py" \
      --video "$video" \
      --object-a-images "$DATA_ROOT/captures/object_a/images" \
      --object-c-image "$DATA_ROOT/captures/object_c/foreground_candidate.png" \
      --fps "$OBJECT_A_FPS" \
      --max-frames "$OBJECT_A_MAX_FRAMES"
  else
    log "Object video not found, expecting images at $DATA_ROOT/captures/object_a/images"
  fi

  if [ -f "$foreground" ]; then
    log "Copying object-C foreground"
    mkdir -p "$DATA_ROOT/captures/object_c"
    cp "$foreground" "$DATA_ROOT/captures/object_c/foreground.png"
  else
    log "Object-C foreground not found, expecting $DATA_ROOT/captures/object_c/foreground.png"
  fi

  if is_low_disk_mode; then
    log "LOW_DISK_MODE: removing temporary unzipped capture input"
    rm -rf "$CAPTURES_INPUT_ROOT"
  fi
}

stage_calvin() {
  local single_split_prepare_args=()
  if is_low_disk_mode; then
    single_split_prepare_args=(--copy)
  fi

  log "Downloading CALVIN LeRobot splits"
  mamba_exec "$ENV_ACT" python "$PROJECT_ROOT/scripts/data/download_calvin_lerobot.py" \
    --target "$DATA_ROOT/datasets/calvin_lerobot" \
    --splits splitA splitB splitC splitD

  log "Preparing splitB"
  mamba_exec "$ENV_ACT" python "$PROJECT_ROOT/scripts/data/prepare_lerobot_dataset.py" \
    --dataset-root "$DATA_ROOT/datasets/calvin_lerobot" \
    --splits splitB \
    --output "$DATA_ROOT/datasets/calvin_prepared/act_env_b" \
    --force \
    "${single_split_prepare_args[@]}"

  log "Preparing splitA+B+C"
  mamba_exec "$ENV_ACT" python "$PROJECT_ROOT/scripts/data/prepare_lerobot_dataset.py" \
    --dataset-root "$DATA_ROOT/datasets/calvin_lerobot" \
    --splits splitA splitB splitC \
    --output "$DATA_ROOT/datasets/calvin_prepared/act_env_abc" \
    --force

  log "Preparing splitD for evaluation"
  mamba_exec "$ENV_ACT" python "$PROJECT_ROOT/scripts/data/prepare_lerobot_dataset.py" \
    --dataset-root "$DATA_ROOT/datasets/calvin_lerobot" \
    --splits splitD \
    --output "$DATA_ROOT/datasets/calvin_prepared/act_env_b_eval_splitD" \
    --force \
    "${single_split_prepare_args[@]}"

  if is_low_disk_mode; then
    log "LOW_DISK_MODE: removing raw CALVIN download after prepared roots were created"
    rm -rf "$DATA_ROOT/datasets/calvin_lerobot"
    stage_cleanup_caches
  fi
}

stage_act() {
  train_act_if_needed \
    "$PROJECT_ROOT/configs/act_env_b.json" \
    "$DATA_ROOT/runs/act/env_b" \
    "splitB"

  train_act_if_needed \
    "$PROJECT_ROOT/configs/act_env_abc.json" \
    "$DATA_ROOT/runs/act/env_abc" \
    "splitA+B+C"

  eval_act_if_needed \
    "$PROJECT_ROOT/configs/act_env_b.json" \
    "$DATA_ROOT/runs/act/env_b_eval_d" \
    "$DATA_ROOT/runs/act/env_b/checkpoints/best" \
    "splitB on splitD"

  eval_act_if_needed \
    "$PROJECT_ROOT/configs/act_env_abc.json" \
    "$DATA_ROOT/runs/act/env_abc_eval_d" \
    "$DATA_ROOT/runs/act/env_abc/checkpoints/best" \
    "splitA+B+C on splitD"
}

stage_prepare_fusion() {
  log "Normalizing fusion asset PLY files"
  mkdir -p "$DATA_ROOT/exports/ply"
  mamba_exec "$ENV_2DGS" python "$PROJECT_ROOT/scripts/vision/prepare_fusion_assets.py" \
    --config "$FUSION_CONFIG" \
    --data-root "$DATA_ROOT" \
    --force
}

stage_merge_fusion() {
  log "Merging normalized fusion PLY files"
  mamba_exec "$ENV_2DGS" python "$PROJECT_ROOT/scripts/vision/merge_scene_assets.py" \
    --config "$FUSION_CONFIG" \
    --output "$FUSION_PLY"
}

stage_vision() {
  ensure_2dgs_ready
  stage_download_mipnerf
  if ! mipnerf_counter_ready "$MIPNERF_COUNTER_DIR"; then
    echo "Missing Mip-NeRF 360 counter dataset: $MIPNERF_COUNTER_DIR" >&2
    echo "Run: bash scripts/cloud_oneclick.sh download-mipnerf" >&2
    return 2
  fi

  log "Training background 2DGS"
  mamba_exec "$ENV_2DGS" python "$PROJECT_ROOT/scripts/vision/run_2dgs.py" train \
    --repo "$REPO_2DGS" \
    --source "$MIPNERF_COUNTER_DIR" \
    --output "$DATA_ROOT/runs/2dgs/background_counter" \
    --resolution "$D2GS_BG_RESOLUTION" \
    --iterations "$D2GS_ITERATIONS"

  log "Rendering background 2DGS"
  mamba_exec "$ENV_2DGS" python "$PROJECT_ROOT/scripts/vision/run_2dgs.py" render \
    --repo "$REPO_2DGS" \
    --source "$MIPNERF_COUNTER_DIR" \
    --output "$DATA_ROOT/runs/2dgs/background_counter"

  log "Running COLMAP for object A"
  mamba_exec "$ENV_2DGS" python "$PROJECT_ROOT/scripts/data/prepare_colmap_object.py" \
    --images "$DATA_ROOT/captures/object_a/images" \
    --workspace "$DATA_ROOT/captures/object_a/colmap"

  log "Training object-A 2DGS"
  mamba_exec "$ENV_2DGS" python "$PROJECT_ROOT/scripts/vision/run_2dgs.py" train \
    --repo "$REPO_2DGS" \
    --source "$DATA_ROOT/captures/object_a/colmap/undistorted" \
    --output "$DATA_ROOT/runs/2dgs/object_a" \
    --resolution "$D2GS_OBJECT_RESOLUTION" \
    --iterations "$D2GS_ITERATIONS"

  log "Generating object B with threestudio"
  mamba_exec "$ENV_AIGC" python "$PROJECT_ROOT/scripts/vision/run_threestudio_asset.py" \
    --repo "$REPO_THREESTUDIO" \
    --prompt "$PROMPT_OBJECT_B" \
    --output "$DATA_ROOT/runs/aigc/object_b_text_to_3d" \
    trainer.max_steps="$THREESTUDIO_MAX_STEPS"

  log "Generating object C with Magic123"
  mamba_exec "$ENV_AIGC" python "$PROJECT_ROOT/scripts/vision/run_magic123_asset.py" \
    --repo "$REPO_MAGIC123" \
    --image "$DATA_ROOT/captures/object_c/foreground.png" \
    --output "$DATA_ROOT/runs/aigc/object_c_image_to_3d" \
    --iters "$MAGIC123_ITERS"

  stage_prepare_fusion
  stage_merge_fusion
}

stage_render_video() {
  if [ ! -f "$FUSION_PLY" ]; then
    echo "Missing fused PLY: $FUSION_PLY" >&2
    echo "Run: bash scripts/cloud_oneclick.sh prepare-fusion && bash scripts/cloud_oneclick.sh render-video" >&2
    return 2
  fi
  log "Rendering fusion walkthrough video"
  mkdir -p "$(dirname "$FUSION_VIDEO")" "$REPORT_ASSETS_DIR"

  renderer="${FUSION_RENDERER:-pointcloud}"
  if [ "$renderer" = "blender" ]; then
    need_cmd "$BLENDER_EXE"
    "$BLENDER_EXE" --background --python "$PROJECT_ROOT/scripts/render/render_fusion_blender.py" -- \
      --input "$FUSION_PLY" \
      --output "$FUSION_VIDEO" \
      --fps 24 \
      --duration "$FUSION_VIDEO_DURATION" \
      --resolution-x "$FUSION_VIDEO_WIDTH" \
      --resolution-y "$FUSION_VIDEO_HEIGHT" \
      --samples "$FUSION_VIDEO_SAMPLES"
  else
    log "Using point-cloud renderer (set FUSION_RENDERER=blender to use Blender)"
    "$PYTHON_EXE" "$PROJECT_ROOT/scripts/render/render_fusion_pointcloud.py" \
      --input "$FUSION_PLY" \
      --output "$FUSION_VIDEO" \
      --fps 24 \
      --duration "$FUSION_VIDEO_DURATION" \
      --resolution-x "$FUSION_VIDEO_WIDTH" \
      --resolution-y "$FUSION_VIDEO_HEIGHT"
  fi

  log "Extracting fusion video keyframes for the report"
  run_ffmpeg -y -i "$FUSION_VIDEO" \
    -vf "fps=1/2" \
    -frames:v 6 \
    "$REPORT_ASSETS_DIR/fusion_keyframe_%02d.png"
}

stage_package() {
  need_cmd "$PYTHON_EXE"
  mkdir -p "$PROJECT_ROOT/results" "$DATA_ROOT/weights"

  if [ -f "$DATA_ROOT/runs/act/env_b/metrics.csv" ] && [ -f "$DATA_ROOT/runs/act/env_abc/metrics.csv" ]; then
    log "Collecting ACT metrics"
    mamba_exec "$ENV_ACT" python "$PROJECT_ROOT/scripts/utils/collect_metrics.py" \
      --inputs "$DATA_ROOT/runs/act/env_b/metrics.csv" "$DATA_ROOT/runs/act/env_abc/metrics.csv" \
      --output-json "$PROJECT_ROOT/results/act_metrics_summary.json" \
      --output-md "$PROJECT_ROOT/results/act_metrics_summary.md"
  else
    log "Metrics CSV files not found yet; skipping metric summary"
  fi

  log "Packaging checkpoints and fusion exports"
  "$PYTHON_EXE" "$PROJECT_ROOT/scripts/utils/package_weights.py" \
    --output "$DATA_ROOT/weights/cvfinal_best_weights.zip" \
    --strict \
    --inputs "$DATA_ROOT/runs/act/env_b/checkpoints/best" \
      "$DATA_ROOT/runs/act/env_abc/checkpoints/best" \
      "$DATA_ROOT/exports/fusion"

  log "Packaging final server artifact bundle"
  mkdir -p "$DATA_ROOT/exports/videos" "$REPORT_ASSETS_DIR"
  "$PYTHON_EXE" "$PROJECT_ROOT/scripts/utils/package_weights.py" \
    --output "$SERVER_BUNDLE" \
    --strict \
    --inputs "$DATA_ROOT/weights/cvfinal_best_weights.zip" \
      "$DATA_ROOT/exports/fusion" \
      "$DATA_ROOT/exports/ply" \
      "$DATA_ROOT/exports/videos" \
      "$REPORT_ASSETS_DIR" \
      "$DATA_ROOT/runs/act/env_b_eval_d" \
      "$DATA_ROOT/runs/act/env_abc_eval_d" \
      "$PROJECT_ROOT/results"
}

stage_audit() {
  log "Auditing final server artifacts"
  need_cmd "$PYTHON_EXE"
  "$PYTHON_EXE" "$PROJECT_ROOT/scripts/utils/final_server_audit.py" \
    --project-root "$PROJECT_ROOT" \
    --data-root "$DATA_ROOT" \
    --fusion-config "$FUSION_CONFIG"
}

stage_disk_usage() {
  print_disk_usage
}

stage_cleanup_caches() {
  log "Cleaning package/download caches and temporary capture input"
  if command -v "$MAMBA_EXE" >/dev/null 2>&1; then
    "$MAMBA_EXE" clean -a -y || true
  fi
  rm -rf \
    "$CONDA_PKGS_DIRS" \
    "$PIP_CACHE_DIR" \
    "$DATA_ROOT/datasets/calvin_lerobot/.cache" \
    "$CAPTURES_INPUT_ROOT"
  mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR"
  if [ "${CLEAN_MODEL_CACHE:-0}" = "1" ]; then
    rm -rf "$HF_HOME" "$TORCH_HOME"
    mkdir -p "$HF_HOME" "$TORCH_HOME"
  fi
  print_disk_usage
}

stage_smoke() {
  need_cmd "$PYTHON_EXE"
  "$PYTHON_EXE" "$PROJECT_ROOT/scripts/setup/run_smoke_tests.py"
}

stage_server_final() {
  stage_prepare_fusion
  stage_merge_fusion
  stage_render_video
  stage_package
  stage_audit
}

stage_finish() {
  stage_act
  stage_vision
  stage_render_video
  stage_package
  stage_audit
}

stage_all() {
  stage_init
  stage_setup_envs
  stage_clone_third_party
  stage_prepare_inputs
  stage_calvin
  stage_act
  stage_vision
  stage_render_video
  stage_package
  stage_audit
}

main() {
  cd "$PROJECT_ROOT"
  local stage="${1:-}"
  case "$stage" in
    all) stage_all ;;
    init) stage_init ;;
    setup-envs) stage_setup_envs ;;
    clone-third-party) stage_clone_third_party ;;
    prepare-inputs) stage_prepare_inputs ;;
    calvin) stage_calvin ;;
    act) stage_act ;;
    finish) stage_finish ;;
    download-mipnerf) stage_download_mipnerf ;;
    wait-and-finish) stage_wait_and_finish ;;
    vision) stage_vision ;;
    prepare-fusion) stage_prepare_fusion; stage_merge_fusion ;;
    render-video) stage_render_video ;;
    package) stage_package ;;
    audit) stage_audit ;;
    server-final) stage_server_final ;;
    disk-usage) stage_disk_usage ;;
    cleanup-caches) stage_cleanup_caches ;;
    smoke) stage_smoke ;;
    -h|--help|help|"") usage ;;
    *) echo "Unknown stage: $stage" >&2; usage; return 2 ;;
  esac
}

main "$@"
