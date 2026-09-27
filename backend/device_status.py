"""Report the compute device visible to the selected Python environment."""
import json


try:
    import torch

    if torch.cuda.is_available():
        backend = "rocm" if torch.version.hip else "cuda"
        result = {"backend": backend, "name": torch.cuda.get_device_name(0)}
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        from accelerate.utils import is_xpu_available
        if is_xpu_available():
            result = {"backend": "xpu", "name": torch.xpu.get_device_name(0)}
        else:
            result = {"backend": "cpu", "name": "CPU", "message": "Intel GPU found, but this PyTorch/Accelerate environment cannot use XPU. Install XPU PyTorch 2.7+ and Accelerate 1.13+."}
    elif torch.backends.mps.is_available():
        result = {"backend": "mps", "name": "Metal GPU"}
    else:
        result = {"backend": "cpu", "name": "CPU", "message": "No GPU is available to PyTorch"}
except Exception as exc:
    result = {"backend": "unavailable", "name": "Compute environment unavailable", "message": str(exc)}

print(json.dumps(result))
