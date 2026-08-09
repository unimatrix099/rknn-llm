# Gemma-4-E4B → RKLLM: conversion handover & context

Working notes for converting `gemma-4-E4B-it` to RKLLM format (w8a8, RK3588)
and benchmarking it against the llama.cpp-based NPU stack on an Orange Pi 5
Ultra. Written as a handover: a machine with more RAM (or an NVIDIA GPU) is
required to finish the conversion — everything else is already prepared.

## Goal

Benchmark Gemma-4-E4B on the official RKLLM runtime (NPU) and compare with
the same model running on the community `rk-llama.cpp` fork, which we have
already measured and patched extensively (see
`github.com/unimatrix099/rk-llama.cpp`, branches `fix/w4a4-calibration-crashes`
and `feat/mixed-precision-pipelines`, plus `docs/backend/RKNPU2-*.md` there).

## Context: what we established so far

### RKLLM vs rk-llama.cpp in one paragraph

RKLLM compiles the whole transformer into a proprietary NPU graph (attention,
norms, KV cache all on-NPU) and is typically ~4x faster at decode than the
matmul-offload approach of rk-llama.cpp (reported Qwen2.5-1.5B: ~19.5 t/s vs
~5 t/s measured on the fork). The trade: closed source, per-model offline
conversion, curated model list, w8a8-only on RK3588 (w4a16 requires RK3576 —
this matches our independent finding that the RK3588 matmul API rejects all
mixed-precision dtypes).

### Reference numbers to compare against

| Config (Gemma-4-E4B Q4_0, RK3588) | pp128 t/s | tg64 t/s |
|---|---|---|
| rk-llama.cpp, NPU W8A8 | 39.8 | 3.3 |
| rk-llama.cpp, CPU only | 25.2 | 4.9 |
| rk-llama.cpp, NPU + `RKNPU_CPU_DECODE=32` | 41.3 | 4.9 |

Rockchip's own `benchmark.md` (this repo) lists Gemma4 **E2B** on RK3588
w8a8 at **11.12 t/s** decode (128 in / 64 out) — E4B is absent from their
table. A community pre-converted E2B exists at
`huggingface.co/Mojo24x7/gemma-4-E2B-it-rkllm-rk3588` (w8a8, ctx4096) and is
useful for validating the runtime setup before the E4B conversion is done.

### Why the conversion needs a big machine (measured, not speculated)

- E4B is physically **7.46 B parameters** ("effective 4B" refers to
  Google-stack memory tricks that don't apply elsewhere).
- The RKLLM-Toolkit **forces float32 when converting on CPU**. Both
  half-precision requests are rejected at runtime with explicit warnings:
  `WARNING: The cpu device not support bfloat16! switch to float32!` and
  the same for float16.
- Peak RAM ≈ 7.46 B × 4 bytes ≈ **30 GB** + torch/quantizer overhead.
  Two attempts on a 31 GB RAM machine (no swap, no GPU) were OOM-killed
  during model load, dying silently as zombies (no Python traceback — check
  `dmesg` for the OOM killer if this happens to you).
- With CUDA (`device='cuda'`), the toolkit keeps fp16 → ~16 GB VRAM+RAM
  footprint instead.
- The toolkit cannot be patched to stream per-tensor: the wheel contains
  33 compiled `.so` modules and only 5 thin `.py` wrappers — all logic
  (loader, quantizer, exporter) is closed source. Only the RKNPU kernel
  driver is open in the whole RKLLM stack.

**Requirements for the conversion machine:** Linux x86_64, Python 3.9–3.12,
either ≥40 GB RAM (CPU path, slow) or an NVIDIA GPU with ≥20 GB VRAM
(`device='cuda'`, faster). ~40 GB free disk (16 GB model + ~8 GB output +
~7 GB venv).

## Exact conversion instructions

### 1. Clone this fork and set up the toolkit

```sh
git clone https://github.com/unimatrix099/rknn-llm.git
cd rknn-llm

python3 -m venv venv            # if ensurepip is missing: python3 -m venv --without-pip venv
                                # then: curl -sL https://bootstrap.pypa.io/get-pip.py | ./venv/bin/python
# pick the wheel matching your python3 minor version (cp39/cp310/cp311/cp312):
./venv/bin/pip install rkllm-toolkit/packages/rkllm_toolkit-1.3.0-cp311-cp311-linux_x86_64.whl
./venv/bin/python -c "from rkllm.api import RKLLM; print('toolkit OK')"
```

The install pulls torch 2.6 + transformers (~6 GB).

### 2. Download the model (ungated mirror, ~16 GB)

```sh
mkdir gemma-4-E4B-it && cd gemma-4-E4B-it
for f in config.json generation_config.json tokenizer.json tokenizer_config.json processor_config.json model.safetensors; do
  curl -L -O "https://huggingface.co/unsloth/gemma-4-E4B-it/resolve/main/$f"
done
cd ..
```

(`unsloth/gemma-4-E4B-it` mirrors `google/gemma-4-E4B-it` without the HF
license gate. Use the official repo with a token if you prefer.)

### 3. Conversion script

Save as `convert_e4b.py` (adjust `modelpath`; set `device='cuda'` +
`dtype='float16'` if you have a GPU, otherwise leave `cpu` — it will run in
float32 and needs the RAM budget above):

```python
from rkllm.api import RKLLM

modelpath = './gemma-4-E4B-it'
llm = RKLLM()

ret = llm.load_huggingface(model=modelpath, model_lora=None, device='cpu',
                           dtype='float32', custom_config=None, load_weight=True)
if ret != 0:
    raise SystemExit('load failed')

ret = llm.build(do_quantization=True, optimization_level=1,
                quantized_dtype='W8A8', quantized_algorithm='normal',
                target_platform='RK3588', num_npu_core=3,
                extra_qparams=None, dataset=None, hybrid_rate=0,
                max_context=4096)
if ret != 0:
    raise SystemExit('build failed')

ret = llm.export_rkllm('./gemma-4-E4B-it_W8A8_RK3588.rkllm')
if ret != 0:
    raise SystemExit('export failed')
print('DONE')
```

```sh
PYTHONUNBUFFERED=1 ./venv/bin/python convert_e4b.py 2>&1 | tee convert.log
```

Notes:
- `quantized_algorithm='normal'` is Rockchip's recommendation for w8a8
  (`grq` is for w4a16, which RK3588 does not support anyway).
- `dataset=None` skips calibration-dataset quantization; for better quality
  a small prompt/response JSON can be provided (see
  `examples/rkllm_api_demo/export/export_rkllm.py` and `data_quant.json`).
- Expected output: `gemma-4-E4B-it_W8A8_RK3588.rkllm`, ~8 GB.
- If the process dies with no traceback: it was the OOM killer.

### 4. Deploy to the board

The board side is **already prepared** (Orange Pi 5 Ultra, Armbian trixie):
this repo is cloned at `~/rknn-llm` and the demo binary is built at
`~/rknn-llm/examples/rkllm_api_demo/deploy/build/llm_demo` (built natively —
the repo's `build-linux.sh` assumes an x86 cross-compiler; on the board just
run cmake directly in `deploy/`).

```sh
scp gemma-4-E4B-it_W8A8_RK3588.rkllm <user>@<board-ip>:~/models/
```

### 5. Run and benchmark on the board

```sh
# stable clocks for reproducible numbers (script from this repo)
sudo bash ~/rknn-llm/scripts/fix_freq_rk3588.sh

export LD_LIBRARY_PATH=~/rknn-llm/rkllm-runtime/Linux/librkllm_api/aarch64
export RKLLM_LOG_LEVEL=1     # prints prefill/generate speed + memory per turn

~/rknn-llm/examples/rkllm_api_demo/deploy/build/llm_demo \
    ~/models/gemma-4-E4B-it_W8A8_RK3588.rkllm 64 4096
# usage: llm_demo model_path max_new_tokens max_context_len
```

`llm_demo` is interactive — type a prompt, and with `RKLLM_LOG_LEVEL=1`
each turn reports prefill t/s and generate t/s. For comparability with the
numbers in this document use a ~128-token prompt and 64 new tokens
(Rockchip's benchmark methodology, matching `llama-bench -p 128 -n 64`).

Also useful: `~/rknn-llm/scripts/eval_perf_watch_npu.sh` and
`eval_perf_watch_cpu.sh` for utilization during inference.

### 6. What to record

- prefill t/s, generate t/s (RKLLM log)
- memory (RKLLM log reports it; also `free -h` on the board)
- subjective output quality vs the fork (the fork's E4B W8A8 runs the same
  weights at 8-bit, so quality should be comparable; w8a8 activation
  quantization differs)

Compare against the reference table above. The interesting question: does
RKLLM's whole-graph NPU execution beat the fork's best (41.3 / 4.9) on this
"chatty"-architecture model (per-layer embeddings, AltUp, LAUREL, shared-KV
ISWA), or does Gemma-4's structure erode RKLLM's usual ~4x decode advantage?
Rockchip's own E2B number (11.12 t/s decode) suggests substantial headroom
over the fork.
