"""Report the compute device visible to the selected Python environment."""
import json


try:
    import torch

    if torch.cuda.is_available():
        backend = "rocm" if torch.version.hip else "cuda"
        result = {"backend": backend, "name": torch.cuda.get_device_name(0)}
    elif torch.backends.mps.is_available():
        result = {"backend": "mps", "name": "Metal GPU"}
    else:
        result = {"backend": "cpu", "name": "CPU", "message": "No GPU is available to PyTorch"}
except Exception as exc:
    result = {"backend": "unavailable", "name": "PyTorch unavailable", "message": str(exc)}

print(json.dumps(result))
