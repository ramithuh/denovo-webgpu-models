# EvoDiff OA_DM_38M WebGPU verification

Complete standalone port through all 16 ByteNet blocks, final `[L,31]` logits, unconditional generation, and motif inpainting. All requested tests passed with zero skips on NVIDIA GB10, driver 580.159.03.

| Precision | Bundle bytes | Worst final-logit relRMS vs matching PyTorch oracle | Warm L32 / L128 / L256 ms |
|---|---:|---:|---|
| FP32 | 151,582,116 | 9.797030612374514e-7 | 10.700 / 10.300 / 12.200 |
| FP16 | 76,004,036 | 9.968881156837651e-7 | 11.400 / 11.900 / 13.900 |
| INT8 | 38,356,136 | 1.0424988532897845e-6 | 17.100 / 18.900 / 19.300 |

Worst CPU final-logit relRMS: `9.202799456566418e-7`. Worst GPU intermediate trace relRMS: `2.7409106690639124e-6`. Each backend validates 300 Mode A trace tensors and 96 complete sampling-step logit matrices across three precisions. A 257-position case exercises every tap at the maximum dilation.

All precisions reproduce Mode B `MELTDCTSLQELFSSCKDPGTSVH` and Mode C `MLTAACDETSSG`, preserving `ACDE`. Quantization error against original FP32 is measured separately: maximum FP16 logit relRMS `0.00018966594991113298`, INT8 `0.007060655695094266`.

Normal, error, initialization-cancellation, and in-flight cancellation paths release transient buffers; destruction leaves zero tracked bytes. Module Worker inference, Cache API/OPFS offline hits, simulated Private Mode fallback, seven Naga-validated WGSL variants, static demo interactions, and vendor sync pass.

| Final command | Exit | Seconds |
|---|---:|---:|
| `npm test` | 0 | 135.96436608489603 |
| `npm run test:cpu` | 0 | 97.7201558239758 |
| `npm run test:gpu` | 0 | 36.110793945379555 |

Evidence: [README and mode matrix](README_WEBGPU.md), [export metrics](fixtures/export_metrics.json), [CPU metrics](fixtures/cpu_metrics.json), [GPU metrics](fixtures/gpu_metrics.json), [WGSL validation](fixtures/wgsl_validation.json), [command records](fixtures/test_runs.json). Full logs and binary oracles are in gitignored `.cache/`; exported weights remain gitignored in `weights/oa_dm_38m/`.

The hypothesis that the complete WGSL implementation matches the pinned matching-precision PyTorch network within `relRMS < 1e-4` and replays deterministic sequences exactly is supported by these fixtures. Next integration step: select weight hosting and use the documented SequenceSet/ResidueTrack mapping. No git commit, push, or Hugging Face upload was performed.

Launchpad submission was attempted with agent session `01a200fd-455a-7103-89d2-5f7316d20f2c`. It returned `project_required`; the connected account lists only `xymol-flow-map`, `xymol-flow-matching`, and `xymol-genie3-port`. This repository has no registered project, so no remote report ID was created. Runs were local harness processes, with no Launchpad job IDs. Submit this report once the appropriate project is registered or supplied.
