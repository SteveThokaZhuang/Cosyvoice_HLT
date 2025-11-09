#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch TTS for CosyVoice2 — single GPU, no text splitting, multi-instance, vLLM/TRT/JIT ready
"""
import argparse, os, sys, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Thread, Lock
import queue
import inspect
import torch, torchaudio
try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
sys.path.append(f"{str(ROOT_DIR)}/third_party/Matcha-TTS")

try:
    from vllm import ModelRegistry
    from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
    try:
        ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)
        print("[vLLM] ModelRegistry registered CosyVoice2ForCausalLM")
    except Exception:
        pass
except Exception:
    pass

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

def parse_bool(x: str) -> bool:
    return str(x).lower() in {"1", "true", "t", "yes", "y"}

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def read_lines(path: Path, start: int, end: int):
    with open(path, "r", encoding="utf-8") as f:
        for idx, raw in enumerate(f):
            if idx < start: continue
            if end >= 0 and idx >= end: break
            line = raw.strip()
            yield idx, line

def save_audio(path: Path, waveform: torch.Tensor, sample_rate: int):
    wf = waveform.detach().to("cpu", non_blocking=True)
    safe_mkdir(path.parent)
    torchaudio.save(str(path), wf, sample_rate)

def worker_thread(worker_id: int, args, prompt_speech, task_queue: "queue.Queue",
                  io_pool: ThreadPoolExecutor, counter, counter_lock, futures_list):
    vllm_kwargs = dict(
        tensor_parallel_size=args.vllm_tp,
        gpu_memory_utilization=args.vllm_gpu_mem,
        max_model_len=args.vllm_max_len,
    )
    # print(f"[Init] worker#{worker_id} loading model (jit={args.jit}, trt={args.trt}, vllm={args.vllm}) ...")
    # try:
    #     print(f"[Init] vLLM kwargs = {vllm_kwargs}")
    #     cosyvoice = CosyVoice2(
    #         args.model_dir,
    #         load_jit=args.jit,
    #         load_trt=args.trt,
    #         load_vllm=args.vllm,
    #         fp16=args.fp16,
    #         vllm_kwargs=vllm_kwargs
    #     )
    # except TypeError as e:
    #     print("[Init][WARN] CosyVoice2(...) does not accept vllm_kwargs on this version:", e)
    #     cosyvoice = CosyVoice2(
    #         args.model_dir,
    #         load_jit=args.jit,
    #         load_trt=args.trt,
    #         load_vllm=args.vllm,
    #         fp16=args.fp16
    #     )
    print(f"[Init] vLLM kwargs = {vllm_kwargs}")
    sig = inspect.signature(CosyVoice2.__init__)
    can_split = all(k in sig.parameters for k in ["vllm_gpu_mem", "vllm_max_len", "vllm_tp"])

    try:
        if can_split:
            print("[Init] This CosyVoice2 version exposes vLLM config; applying user settings.")
            cosyvoice = CosyVoice2(
                args.model_dir,
                load_jit=args.jit, load_trt=args.trt, load_vllm=args.vllm, fp16=args.fp16,
                vllm_gpu_mem=args.vllm_gpu_mem,
                vllm_max_len=args.vllm_max_len,
                vllm_tp=args.vllm_tp,
                vllm_enforce_eager=False
            )
        else:
            print("[Init][WARN] This CosyVoice2 version exposes no vLLM config; using library defaults.")
            cosyvoice = CosyVoice2(
                args.model_dir,
                load_jit=args.jit, load_trt=args.trt, load_vllm=args.vllm, fp16=args.fp16,vllm_enforce_eager=False
            )
    except TypeError as e:
        print("[Init][ERROR] Unexpected CosyVoice2 signature; falling back without vLLM tuning:", e)
        cosyvoice = CosyVoice2(
            args.model_dir,
            load_jit=args.jit, load_trt=args.trt, load_vllm=args.vllm, fp16=args.fp16
        )
        print(f"[Init] worker#{worker_id} model ready.")

    if args.warmup > 0:
        warmup_text = args.warmup_text or "你好，世界。"
        for _ in range(args.warmup):
            try:
                try:
                    gen = cosyvoice.inference_zero_shot(
                        warmup_text, args.prompt_text, prompt_speech,
                        stream=False, text_frontend=args.text_frontend
                    )
                except TypeError:
                    gen = cosyvoice.inference_zero_shot(
                        warmup_text, args.prompt_text, prompt_speech,
                        stream=False
                    )
                _ = list(gen)
            except Exception as e:
                print(f"[Warmup] worker#{worker_id} warmup failed: {e}")
                break
        print(f"[Warmup] worker#{worker_id} done.")

    with torch.inference_mode():
        while True:
            try:
                item = task_queue.get(timeout=2.0)
            except queue.Empty:
                break
            if item is None:
                break
            idx, text, out_path = item
            try:
                try:
                    gen = cosyvoice.inference_zero_shot(
                        text, args.prompt_text, prompt_speech,
                        stream=args.stream, text_frontend=args.text_frontend
                    )
                except TypeError:
                    gen = cosyvoice.inference_zero_shot(
                        text, args.prompt_text, prompt_speech,
                        stream=args.stream
                    )

                if args.stream:
                    chunks = []
                    sr = cosyvoice.sample_rate
                    for pack in gen:
                        chunks.append(pack["tts_speech"].detach().cpu())
                    if not chunks:
                        continue
                    audio = torch.cat(chunks, dim=-1)
                else:
                    packs = list(gen)
                    if not packs:
                        continue
                    audio = packs[-1]["tts_speech"].detach().cpu()
                    sr = cosyvoice.sample_rate

                fut = io_pool.submit(save_audio, out_path, audio, sr)
                futures_list.append(fut)

                with counter_lock:
                    counter["done"] += 1
                    d = counter["done"]
                    if d % 50 == 0:
                        print(f"[Prog] worker#{worker_id} -> {d} utterances queued for save")

            except Exception as e:
                print(f"[Error] worker#{worker_id} failed on line {idx}: {e}")
            finally:
                task_queue.task_done()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=str, required=True)
    p.add_argument("--prompt-wav", type=str, required=True)
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--prompt-text", type=str, default="")
    p.add_argument("--prefix", type=str, default="utt_")
    p.add_argument("--suffix", type=str, default=".wav")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=-1)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--instances", type=int, default=1)
    p.add_argument("--fp16", type=parse_bool, default=True)
    p.add_argument("--text-frontend", type=parse_bool, default=True)
    p.add_argument("--stream", type=parse_bool, default=False)
    p.add_argument("--vllm", type=parse_bool, default=True)
    p.add_argument("--trt", type=parse_bool, default=False)
    p.add_argument("--jit", type=parse_bool, default=False)
    p.add_argument("--vllm-tp", type=int, default=1)
    p.add_argument("--vllm-gpu-mem", type=float, default=0.9)
    p.add_argument("--vllm-max-len", type=int, default=8192)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--warmup-text", type=str, default="")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Using device: {device}")
    if device.type != "cuda":
        print("[Warning] CUDA not available. This script is designed for single-GPU use.")

    outdir = Path(args.outdir); safe_mkdir(outdir)
    prompt_speech = load_wav(args.prompt_wav, 16000)

    q = queue.Queue(maxsize=4 * max(1, args.instances))
    io_pool = ThreadPoolExecutor(max_workers=max(1, args.workers))
    futures = []
    counter, counter_lock = {"done": 0}, Lock()

    instances = max(1, int(args.instances))
    threads = []
    t0 = time.time()
    for wid in range(instances):
        th = Thread(target=worker_thread,
                    args=(wid, args, prompt_speech, q, io_pool, counter, counter_lock, futures),
                    daemon=True)
        th.start()
        threads.append(th)

    total_tasks = 0
    for idx, text in read_lines(Path(args.input), args.start, args.end):
        if not text: continue
        out_path = outdir / f"{args.prefix}{idx:07d}{args.suffix}"
        if args.resume and out_path.exists() and not args.overwrite: continue
        q.put((idx, text, out_path))
        total_tasks += 1

    q.join()
    for _ in range(instances): q.put(None)
    for th in threads: th.join()

    for fut in as_completed(futures): _ = fut.result()
    io_pool.shutdown(wait=True)

    elapsed = time.time() - t0
    print(f"[Done] Wrote {counter['done']} / {total_tasks} utterances to {str(outdir)} in {elapsed/60:.2f} min "
          f"({(counter['done'] / elapsed) if elapsed > 0 else 0:.2f} utt/s).")

if __name__ == "__main__":
    main()
