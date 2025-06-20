# GPT-2 XL Quantization Benchmark

## 1. Overview  
This short project compares three variants of GPT-2 XL on CPU.
Initially, GPU monitoring was attempted, but due to several compatibility issues,
I had to fall back to CPU monitoring.
The three variants used are the following:

1. **baseline**: full-precision FP32  
2. **dynamic**: PyTorch dynamic quantization (only `Linear` → INT8)  
3. **onnx**: end-to-end ONNX export + dynamic quantization (all weights INT8)

Measured:

- **Model size** on disk  
- **Throughput** (tokens/sec) for a 100-token generation  
- **Total latency** (seconds for 100 tokens)  
- **Time-to-first-token** (TTFT)  
- **Peak & 90th-percentile** RSS memory (MB)  
- **Peak & 90th-percentile** CPU utilization (%)

---

## 2. Setup  
- Native Linux, Python 3.12, PyTorch, Transformers, ONNX Runtime, `psutil`.  
- Single thread, CPU only (`device="cpu"`).  
- Fixed prompt (“Once upon a time, there was a magical forest”) and `max_new_tokens=100`.

---

## 3. Key Results

| Variant   | Size (MB) | TPS  | Total (s) | TTFT (s) | Peak Mem (MB) | 90% Mem (MB) | Peak CPU (%) | 90% CPU (%) |
|:---------:|:---------:|:----:|:---------:|:--------:|:-------------:|:------------:|:------------:|:-----------:|
| baseline  | 5942      | 5.52 | 18.12     | 0.35     | 6788          | 6783         | 56.2         | 52.7        |
| dynamic   | 6019      | 5.74 | 17.44     | 0.35     | 6963          | 6956         | 57.3         | 52.3        |
| onnx      | 1568      | 7.28 | 13.74     | 0.06     | 10056         | 10055        | 100          | 100         |

*(Values rounded.)*

---

## 4. Analysis of Results

- **Model size**  
  - Dynamic quant barely shrinks the `.pth` (only `Linear` layers)—metadata bumps size slightly.  
  - ONNX-quant is true 8-bit for *all* weights ~1/4 the FP32 footprint.  

- **Speed & latency**  
  - Dynamic quant yields only a few % FPS gain on CPU (same TTFT).  
  - ONNX quant + optimized runtime gives ~32 % higher throughput and a ~6× drop in TTFT.  

- **Memory & CPU util**  
  - Baseline/dynamic each use ~6.8 GB RSS and ~50–60 % CPU.  
  - ONNX loads a bigger graph/executor ⇒ ~10 GB RSS and maxes CPU at 100 %.

---

## 5. Next Steps

1. **Tidy measurements**  
   - Restart or clear memory between runs so each variant’s RSS is isolated.  
   - Repeat benchmarks 3× and report mean ± std.

2. **Static quant**  
   - Apply post-training static/Fx quantization (all weights INT8) in PyTorch—should shrink `.pth` and reduce peak memory, with a modest speedup.

3. **GPU quant**  
   - Install `onnxruntime-gpu` and rerun ONNX-quant on CUDA.  
   - Explore FX/AOT quant paths in PyTorch that generate CUDA quant kernels.

4. **FP16 baseline**  
   - Benchmark `model.half()` on GPU (or CPU with bfloat16) for a 16-bit comparison.

5. **Containerise for cross-platform**
    - Perform the benchmark inside a Docker container.
    - Look into nvidia/cuda (GPU) and python:slim (CPU) images.

---
