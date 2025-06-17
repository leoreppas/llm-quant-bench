import torch
import time
import threading
import pynvml
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Dict, Any

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


if __name__ == "__main__":
    # Run baseline benchmark
    gpu_monitor = GPUMonitor()
    gpu_monitor.start()

    model = AutoModelForCausalLM.from_pretrained("gpt2-xl")
    tokenizer = AutoTokenizer.from_pretrained("gpt2-xl")
    prompt = "Once upon a time, there was a magical forest"

    base = benchmark_single_prompt(
        model=model,
        tokenizer=tokenizer,
        input_prompt_text=prompt,
        temperature=0.7,
        top_p=0.95,
        max_new_tokens=100
    )

    gpu_monitor.stop()
    base["peak_usage"] = gpu_monitor.get_peak_usage()
    base["peak_utilisation"] = gpu_monitor.get_peak_utilisation()
    base["p90_usage"] = gpu_monitor.get_p90_usage()
    base["p90_utilisation"] = gpu_monitor.get_p90_utilisation()

    print(f"Generated: {base['generated_text']}")
    print(f"TTFT: {base['ttft_seconds']:.4f}s | Total: {base['total_time_seconds']:.4f}s | TPS: {base['tokens_per_second']:.2f}")
    print(f"Peak Mem: {base['peak_usage']:.2f} MB | 90th% Mem: {base['p90_usage']:.2f} MB")
    print(f"Peak GPU%: {base['peak_utilisation']:.2f}% | 90th% GPU%: {base['p90_utilisation']:.2f}%")

