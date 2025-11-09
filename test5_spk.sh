python gputest5_spk.py \
  --model-dir ../pretrained_models/CosyVoice2-0.5B \
  --input text0.txt \
  --outdir ref_out \
  --prompt-wav ../asset/zero_shot_prompt.wav \
  --prompt-text "希望你以后能够做的比我还好呦。" \
  --use-spkinfo true \
  --spk-id speaker0_spk

python - <<'PY'
import torch, os, sys
md = "../pretrained_models/CosyVoice2-0.5B"
d = torch.load(os.path.join(md,"spk2info.pt"), map_location="cpu")
print("keys:", list(d.keys())[:10])
print("has speaker0_spk? ->", "speaker0_spk" in d)
PY

python gputest5_spk.py \
  --model-dir ../pretrained_models/CosyVoice2-0.5B \
  --input text.txt \
  --outdir spk_out/ \
  --use-spkinfo true \
  --spk-id speaker0_spk &
iostat -x 5 &