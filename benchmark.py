import os
import torch
import time
import threading
import pynvml
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Dict, Any
import torch.quantization as torch_quant
import onnxruntime.quantization as onnx_quant
import onnxruntime

class GPUMonitor:
    def __init__(self, monitoring_interval: float = 0.1):
        self.monitoring_interval = monitoring_interval
        self._gpu_memory_usage = []
        self._gpu_utilisation = []
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
        self._gpu_memory_usage = []
        self._gpu_utilisation = []
        while self._is_monitoring:
            time.sleep(self.monitoring_interval)
            if not self._is_monitoring:
                break
            handle = self._nvml.nvmlDeviceGetHandleByIndex(0)
            mem_info = self._nvml.nvmlDeviceGetMemoryInfo(handle)
            util     = self._nvml.nvmlDeviceGetUtilizationRates(handle)

            mem = mem_info.used / 1024**2      # MB
            gpu_pct = util.gpu                 # %

            self._gpu_memory_usage.append(mem)
            self._gpu_utilisation.append(gpu_pct)

    def get_peak_usage(self) -> float:
        return max(self._gpu_memory_usage) if self._gpu_memory_usage else 0.0

    def get_peak_utilisation(self) -> float:
        return max(self._gpu_utilisation) if self._gpu_utilisation else 0.0

    def get_p90_usage(self) -> float:
        if not self._gpu_memory_usage:
            return 0.0
        sorted_vals = sorted(self._gpu_memory_usage)
        idx = int(len(sorted_vals) * 0.9)
        return sorted_vals[idx]

    def get_p90_utilisation(self) -> float:
        if not self._gpu_utilisation:
            return 0.0
        sorted_vals = sorted(self._gpu_utilisation)
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
    # Select device
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Set tokenizer padding if missing
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Tokenize input with attention mask
    enc = tokenizer(
        input_prompt_text,
        return_tensors="pt",
        padding=True
    ).to(device)

    # Warm-up
    with torch.no_grad():
        _ = model.generate(**enc, max_new_tokens=1, do_sample=False)

    # Measure total generation time
    if device.startswith("cuda"): torch.cuda.synchronize()
    start_time = time.perf_counter()
    with torch.no_grad():
        output = model.generate(
            **enc,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            do_sample=True
        )
    if device.startswith("cuda"): torch.cuda.synchronize()
    end_time = time.perf_counter()
    total_time = end_time - start_time

    # Measure TTFT (time to first token)
    if device.startswith("cuda"): torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = model.generate(**enc, max_new_tokens=1, do_sample=False)
    if device.startswith("cuda"): torch.cuda.synchronize()
    t1 = time.perf_counter()
    ttft = t1 - t0

    tokens_per_second = max_new_tokens / total_time if total_time > 0 else 0.0
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return {
        "ttft_seconds": ttft,
        "total_time_seconds": total_time,
        "tokens_per_second": tokens_per_second,
        "generated_text": generated_text
    }


def run_benchmark(model_name: str, prompt: str, quant: Optional[str] = None) -> Dict[str, Any]:
    # create per-variant folder
    variant_name = quant or "baseline"
    os.makedirs(variant_name, exist_ok=True)

    # Load base model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Save baseline model weights for size measurement
    baseline_path = os.path.join(variant_name, "baseline.pth")
    torch.save(model.state_dict(), baseline_path)
    size_mb = os.path.getsize(baseline_path) / 1024**2

    # Apply PyTorch dynamic quant if var is set
    device_override: Optional[str] = None
    if quant == "dynamic":
        model = torch_quant.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        # Save dynamic model weights for size measurement
        dyn_path = os.path.join(variant_name, "dynamic.pth")
        torch.save(model.state_dict(), dyn_path)
        size_mb = os.path.getsize(dyn_path) / 1024**2
        device_override = "cpu"

    # Export and quantize ONNX if var is set
    onnx_path: Optional[str] = None
    if quant == "onnx":
        # Export to ONNX
        enc = tokenizer(prompt, return_tensors="pt")
        onnx_file = os.path.join(variant_name, "model.onnx")
        torch.onnx.export(
            model.cpu(), enc["input_ids"],
            onnx_file,
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                          "logits":   {0: "batch", 1: "seq"}},
            opset_version=14
        )
        # Quantize the ONNX model
        quantized_path = os.path.join(variant_name, "model-quant.onnx")
        onnx_quant.quantize_dynamic(
            onnx_file, quantized_path,
            weight_type=onnx_quant.QuantType.QInt8
        )
        onnx_path = quantized_path
        # Measure ONNX-quant file size
        size_mb = os.path.getsize(quantized_path) / 1024**2

    # Start GPU monitoring
    gpu_monitor = GPUMonitor()
    gpu_monitor.start()

    # Run inference and timing
    if quant != "onnx":
        metrics = benchmark_single_prompt(
            model=model,
            tokenizer=tokenizer,
            input_prompt_text=prompt,
            temperature=0.7,
            top_p=0.95,
            max_new_tokens=100,
            device=device_override
        )
    else:
        # ONNXRuntime path
        providers = (["CUDAExecutionProvider"]
                     if "CUDAExecutionProvider" in onnxruntime.get_available_providers()
                     else ["CPUExecutionProvider"])
        sess = onnxruntime.InferenceSession(onnx_path, providers=providers)
        inputs = tokenizer(prompt, return_tensors="np")
        onnx_inputs = {"input_ids": inputs["input_ids"]}

        # Warm-up run
        _ = sess.run(None, onnx_inputs)
        # Measure ONNX inference time
        if "CUDAExecutionProvider" in providers and torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        _ = sess.run(None, onnx_inputs)
        if "CUDAExecutionProvider" in providers and torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()
        metrics = {
            "ttft_seconds":       None,
            "total_time_seconds": end - start,
            "tokens_per_second":  1.0 / (end - start),
            "generated_text":     None
        }

    # Stop GPU monitoring and attach resource stats
    gpu_monitor.stop()
    metrics.update({
        "peak_usage_mb": gpu_monitor.get_peak_usage(),
        "peak_util_pct": gpu_monitor.get_peak_utilisation(),
        "p90_usage_mb":  gpu_monitor.get_p90_usage(),
        "p90_util_pct":  gpu_monitor.get_p90_utilisation()
    })

    # Record model file size
    metrics["model_size_mb"] = size_mb

    # Print & return
    print(f"Variant: {variant_name}")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    return metrics


results = {}
for variant in [None, "dynamic", "onnx"]:
    print(f"-------------{variant or 'baseline'}-------------")
    results[variant or "baseline"] = run_benchmark(
        model_name="gpt2-xl",
        prompt="Once upon a time, there was a magical forest",
        quant=variant
    )
