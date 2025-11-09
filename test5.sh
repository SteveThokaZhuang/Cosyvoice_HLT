# 建议先把编译/缓存迁到本地内存盘，避免 NFS 卡顿
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256
export CUDA_DEVICE_MAX_CONNECTIONS=32
export TRITON_CACHE_DIR=/dev/shm/triton_cache_$USER
export XDG_CACHE_HOME=/dev/shm/xdg_$USER
export HF_HOME=/dev/shm/hf_$USER
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export TORCHINDUCTOR_CACHE_DIR=/dev/shm/ti_$USER
export CUDA_CACHE_PATH=/dev/shm/cuda_$USER
mkdir -p "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" "$HF_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH"
# export VLLM_DISABLE_MEMORY_PROFILING=1
export VLLM_DISABLE_MEMORY_PROFILING=1 && export VLLM_USE_META_MODEL=0

CUDA_VISIBLE_DEVICES=0 python gputest5.py \
  --model-dir ../pretrained_models/CosyVoice2-0.5B \
  --prompt-wav ../asset/zero_shot_prompt.wav \
  --input text.txt \
  --outdir tts_out_25 \
  --prompt-text "希望你以后能够做的比我还好呦。" \
  --instances 1 