# Quantized models on sm_75: status and plan

The mainline engine runs three quality tiers. On sm_75 only the bf16 `reference` tier is *available without extra
files* today; the quantized tiers work when the models are already quantized, and the pieces needed to make them
first-class (and to produce the files) are listed below with a plan.

## What the tiers mean

| Tier | Weights | Attention | What the loaded DiT must be |
|---|---|---|---|
| `reference` | bf16, exact | exact | a bf16 DiT and text encoder |
| `high` | INT8 ConvRot (W8A8), optional quantized text encoder | 8-bit (`q8_fp16`/`q8_fp8`) | an INT8 ConvRot DiT **file**, or a bf16 DiT *and* 8-bit attention |
| `turbo` | INT4 ConvRot (`convrot_w4a4`) or NVFP4 | 8-bit | an INT4/NVFP4 DiT file (NVFP4 needs an RTX 50) |

`planner::resolve_tier` (src/planner/planner.cpp) enforces that: `high` with exact attention on a bf16 DiT is "the
Reference configuration" and is refused, and on sm_75 the missing 8-bit attention kernels make `exact_when_no_q8` pick
exact attention, so `high` needs an INT8 DiT file. `turbo` only accepts a 4-bit file.

## What already works on sm_75

Measured on the RTX 2080 8 GiB with the SFT bundle (`balanced`), this branch, after the attention and VAE work:

| Configuration | DiT file | Denoise, 256^2, 8 steps | Denoise, 1024^2, 2 steps | Device peak |
|---|---|---|---|---|
| `reference` | bf16 bundle | 7.50 s (0.94 s/step) | 8.60 s (4.3 s/step) | 6.37 GiB |
| `high` | INT8 ConvRot, 5.04 GB | **0.99 s (0.12 s/step)** | **5.18 s (2.54 s/step)** | 5.27 GiB |
| `turbo` | INT4 ConvRot, 2.83 GB | 1.14 s (0.14 s/step) | 5.16 s (2.57 s/step) | 3.06 GiB |

- The INT8 GEMMs are CUTLASS Sm75 m8n8k16 kernels with the sm_80 epilogues (dequantize+bias, gated residual,
  SiTU-GLU) under `src/kernels/sm75/int8_gemm_sm75.cu`; one tile configuration (64x128x64) fits Turing's 64 KiB of
  shared memory, so forcing another tile is refused (`check_sm75_tile`).
- INT4 weights stay packed and are unpacked into the INT8 GEMM (`quant4/w4a8_decode.cu`), so `turbo` is correct on
  Turing (about 1.5x the INT8 step in the planner's price, not a native 4-bit MMA).
- Exact attention runs with every tier; the quantized tiers do not need 8-bit attention on sm_75.
- `moss-infer quantize` runs on an 8 GiB Turing card: the 5B DiT became a 5.04 GB INT8 ConvRot file in 49 s
  (252 GEMMs quantized, 0 kept bf16).
- The VAE stays fp32 SIMT either way (see `docs/sm75.md`).

## Gaps

**G1 - no INT4 writer.** `moss-infer quantize --format` accepts only `int8` and `nvfp4`
(src/tools/cli/cmd_quantize.cpp). `convrot_w4a4` files (the only DiT format `turbo` accepts) therefore come from
elsewhere. The pieces to write them exist: `src/weights/quant/convrot_w4a4.{h,cpp}` quantizes W4 with the ConvRot
groups and `quant_group_size`, and the loader already reads the format. The file metadata is a
`_quantization_metadata` JSON holding `format_version` and per-layer
`{"format": "convrot_w4a4", "convrot_groupsize": 256, "quant_group_size": 64}` (checked in a community file).

**G2 - `high` with the bf16 bundle is refused on sm_75.** The tier is defined as INT8 GEMMs *plus* 8-bit attention;
sm_75 has no 8-bit attention, so the resolver falls back to exact attention and then refuses the combination. A
Turing user has to quantize the DiT first (or download an INT8 file) to use `high`.

**G3 - no 8-bit attention kernel.** `q8_fp16`/`q8_fp8` are refused on sm_75 `--attention`. Besides blocking G2, this
is the mainline's long-sequence accelerator: at 1024^2 the DiT's exact attention is 1.05 s per step
(7.3 TFLOPS with the new kernel), which an INT8 QK mainloop should roughly halve.

**G4 - LoRA on quantized DiTs is unverified on sm_75.** Merged adapters on INT8 block GEMMs go through
`merge_requantize_int8` (src/models/lora/lora_merge.cpp); on sm_75 every LoRA is merged (no side branches), and an
INT4 DiT has no merge path at all, so `--tier turbo --lora` should refuse clearly instead of silently ignoring the
adapter. Needs tests and, if missing, a message.

**G5 - the app has to offer it.** MossStudio decides what to download and which tiers to show; a Turing card should be
able to run `quantize` (engine CLI) or fetch INT8/INT4 files, and the tier selector should enable `high`/`turbo` when
one is loaded.

## Plan

**P1 - INT4 output in `quantize` (small, independent).** Add `--format int4`: quantize each block GEMM with
`weights::quant::convrot_w4a4`, write packed W4 weights plus the per-layer metadata above (group 256, quant group 64 by
default, `--group` as for INT8), keep the non-block tensors bf16. Then `moss-infer generate --dit <int4> --tier turbo`
works with a file produced on the same card. Test: the produced file's size (expected ~2.9 GB), `inspect`, a 256^2 and
a 1024^2 generation, and compare its output with the community file's.

**P2 - Turing 8-bit attention (`q8_fp16`).** A new kernel in `src/kernels/sm75/`: INT8 QK with the Turing m8n8k16 MMA
over the ConvRot-quantized q/k (the quantizers are already portable: `quant/rotate_quantize.cu`), fp16 P·V with the
existing m16n8k8 MMA, exact softmax in fp32. Wire it into `attention_supported`, `MossDit`'s attention dispatch and
`resolve_tier` so `--attention q8_fp16` is accepted on cc 7.5 and `high` with a bf16 DiT resolves to it. Expected:
attention 14.3 ms -> 8-9 ms at 1024^2 (0.5 s per step saved) and `high` works without a quantized file.

**P3 - or, if P2 waits: a sm_75 tier rule.** Let `high` on cc 7.5 quantize the block GEMMs at load
(`LoadOptions::int8_block_gemms` currently only follows an INT8 file) and keep exact attention, documented as a
sm_75-specific definition of `high`. Small (planner + load options), but it changes what `high` means on Turing, so
P2 is the cleaner long-term answer.

**P4 - LoRA on quantized tiers.** Test `--tier high --lora` on sm_75 (merge + requantize), and make `--tier turbo
--lora` refuse with a message that names the missing merge path. If the requantize path turns out to need sm_80
kernels, port the host merge (`quant::quantize_convrot_rows` is already used by the sm_80 path).

**P5 - docs and app.** Update `docs/sm75.md`'s limits with the outcome, add the INT4 production step and the measured
tier table here, and tell MossStudio which files/tiers a cc 7.5 device can use.

## Test matrix

- `sm75_check int8` (existing) plus a `quantize` -> load -> generate round trip for INT8 and INT4 at 256^2 and 1024^2.
- `high`/`turbo` with `--attention exact` and (after P2) `q8_fp16`: same prompt and seed, check the report's tier line
  and `quality.gpu_arch`.
- LoRA: `--tier reference` (merged, works today), `--tier high --lora` (P4), `--tier turbo --lora` (clear refusal).
- Regression: `sm75_check` all cases, and the OFF build (`MOSSINFER_SM75=OFF`) still compiles and passes.
