import os
import time
import threading
import numpy as np
import torch
import torch.quantization as torch_quant
import pynvml
import psutil
import onnxruntime
import onnxruntime.quantization as onnx_quant
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Dict, Any
import matplotlib.pyplot as plt


class ResourceMonitor:
    def __init__(self, monitoring_interval: float = 0.1):
        self.monitoring_interval = monitoring_interval
        self._mem_usage = []
        self._cpu_util = []
        self._is_monitoring = False
        pynvml.nvmlInit()
        self._nvml = pynvml

    def start(self):
        print("Started monitoring...")
        self._is_monitoring = True
        self._thread = threading.Thread(target=self._monitor)
        self._thread.start()

    def stop(self):
        self._is_monitoring = False
        self._thread.join()
        print("Stopped monitoring...")
        self._nvml.nvmlShutdown()

    def _monitor(self):
        self._mem_usage = []
        self._cpu_util = []
        while self._is_monitoring:
            time.sleep(self.monitoring_interval)
            if not self._is_monitoring:
                break
            # system memory (RSS) and CPU %
            mem = psutil.Process().memory_info().rss / 1024**2  # MB
            cpu = psutil.cpu_percent(interval=None)
            self._mem_usage.append(mem)
            self._cpu_util.append(cpu)

    def get_peak_memory(self) -> float:
        return max(self._mem_usage) if self._mem_usage else 0.0

    def get_peak_cpu(self) -> float:
        return max(self._cpu_util) if self._cpu_util else 0.0

    def get_p90_memory(self) -> float:
        if not self._mem_usage:
            return 0.0
        sorted_vals = sorted(self._mem_usage)
        idx = int(len(sorted_vals) * 0.9)
        return sorted_vals[idx]

    def get_p90_cpu(self) -> float:
        if not self._cpu_util:
            return 0.0
        sorted_vals = sorted(self._cpu_util)
        idx = int(len(sorted_vals) * 0.9)
        return sorted_vals[idx]


def benchmark_single_prompt(
    model,
    tokenizer,
    input_prompt_text: str,
    temperature: float = 1.0,
    top_p: float = 0.95,
    max_new_tokens: int = 100,
    device: Optional[str] = None
) -> Dict[str, Any]:
    # force CPU-only
    device = "cpu"
    model.to(device)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    enc = tokenizer(
        input_prompt_text,
        return_tensors="pt",
        padding=True
    ).to(device)

    # warm-up
    with torch.no_grad():
        _ = model.generate(**enc, max_new_tokens=1, do_sample=False)

    # total latency
    start_time = time.perf_counter()
    with torch.no_grad():
        output = model.generate(
            **enc,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            do_sample=True
        )
    end_time = time.perf_counter()
    total_time = end_time - start_time

    # TTFT
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = model.generate(**enc, max_new_tokens=1, do_sample=False)
    t1 = time.perf_counter()
    ttft = t1 - t0

    tps = max_new_tokens / total_time if total_time > 0 else 0.0
    gen_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return {
        "ttft_seconds":       ttft,
        "total_time_seconds": total_time,
        "tokens_per_second":  tps,
        "generated_text":     gen_text
    }


def run_benchmark(model_name: str, prompt: str, quant: Optional[str] = None) -> Dict[str, Any]:
    variant = quant or "baseline"
    os.makedirs(variant, exist_ok=True)

    # load on CPU
    model = AutoModelForCausalLM.from_pretrained(model_name).to("cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # save baseline weights
    base_path = os.path.join(variant, "baseline.pth")
    torch.save(model.state_dict(), base_path)
    size_mb = os.path.getsize(base_path) / 1024**2

    if quant == "dynamic":
        model = torch_quant.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        dyn_path = os.path.join(variant, "dynamic.pth")
        torch.save(model.state_dict(), dyn_path)
        size_mb = os.path.getsize(dyn_path) / 1024**2

    onnx_path = None
    if quant == "onnx":
        # ensure pad_token before padding=True
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        enc = tokenizer(prompt, return_tensors="pt", padding=True)
        onnx_f = os.path.join(variant, "model.onnx")
        torch.onnx.export(
            model, enc["input_ids"], onnx_f,
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                          "logits":   {0: "batch", 1: "seq"}},
            opset_version=14
        )
        quant_f = os.path.join(variant, "model-quant.onnx")
        onnx_quant.quantize_dynamic(
            onnx_f, quant_f,
            weight_type=onnx_quant.QuantType.QInt8
        )
        onnx_path = quant_f
        size_mb = os.path.getsize(quant_f) / 1024**2

    monitor = ResourceMonitor()
    monitor.start()

    if quant != "onnx":
        metrics = benchmark_single_prompt(
            model=model,
            tokenizer=tokenizer,
            input_prompt_text=prompt,
            temperature=0.7,
            top_p=0.95,
            max_new_tokens=100
        )
    else:
        sess = onnxruntime.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )
        # initial pad-token assurance already done
        enc_np = tokenizer(prompt, return_tensors="np", padding=True)
        input_ids = enc_np["input_ids"]

        # TTFT
        _ = sess.run(None, {"input_ids": input_ids})
        t0 = time.perf_counter()
        out = sess.run(None, {"input_ids": input_ids})
        t1 = time.perf_counter()
        ttft = t1 - t0

        # loop
        N = 100
        start = time.perf_counter()
        for _ in range(N):
            out = sess.run(None, {"input_ids": input_ids})
            logits = out[0]
            tok = int(np.argmax(logits[0, -1]))
            input_ids = np.concatenate([input_ids, [[tok]]], axis=1)
        end = time.perf_counter()
        gen_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        tps = N / (end - start)

        metrics = {
            "ttft_seconds":       ttft,
            "total_time_seconds": end - start,
            "tokens_per_second":  tps,
            "generated_text":     gen_text
        }

    monitor.stop()
    metrics.update({
        "peak_memory_mb": monitor.get_peak_memory(),
        "peak_cpu_pct":   monitor.get_peak_cpu(),
        "p90_memory_mb":  monitor.get_p90_memory(),
        "p90_cpu_pct":    monitor.get_p90_cpu()
    })
    metrics["model_size_mb"] = size_mb

    print(f"Variant: {variant}")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    return metrics


# run all three
results: Dict[str, Dict[str, Any]] = {}
for v in [None, "dynamic", "onnx"]:
    name = v or "baseline"
    print(f"----- {name} -----")
    results[name] = run_benchmark(
        "gpt2-xl",
        "Once upon a time, there was a magical forest",
        quant=v
    )


# Visualise Results
def plot_metric(metric: str):
    titles = {
        "tokens_per_second":  "Throughput (tokens/sec)",
        "ttft_seconds":       "Time to First Token (s)",
        "total_time_seconds": "Total Latency (s)",
        "model_size_mb":      "Model Size (MB)",
        "peak_memory_mb":     "Peak Memory (MB)",
        "p90_memory_mb":      "90th% Memory (MB)",
        "peak_cpu_pct":       "Peak CPU Utilisation (%)",
        "p90_cpu_pct":        "90th% CPU Utilisation (%)",
    }
    title = titles.get(metric, metric)
    variants = ["baseline", "dynamic", "onnx"]
    vals = [results[v][metric] for v in variants]

    fig, ax = plt.subplots()
    ax.bar(variants, vals)
    ax.set_xlabel("Variant")
    ax.set_ylabel(title)
    ax.set_title(title)
    for i, val in enumerate(vals):
        ax.text(i, val, f"{val:.2f}", ha="center", va="bottom")

    fname = f"{metric}.png"
    fig.savefig(fname, bbox_inches="tight")
    print(f"→ saved {fname}")
    plt.close(fig)


for metric in [
    "tokens_per_second", "ttft_seconds", "total_time_seconds",
    "model_size_mb", "peak_memory_mb", "p90_memory_mb",
    "peak_cpu_pct", "p90_cpu_pct"
]:
    plot_metric(metric)
