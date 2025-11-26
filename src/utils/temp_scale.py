import os, json, torch
from typing import Optional

def load_temperature(json_path: Optional[str]) -> Optional[torch.Tensor]:
    if not json_path or not os.path.exists(json_path):
        return None
    with open(json_path) as f:
        T = float(json.load(f)["T"])
    return torch.tensor(T, dtype=torch.float32)

def apply_temperature(logits: torch.Tensor, T: Optional[torch.Tensor]) -> torch.Tensor:
    if T is None:
        return logits
    return logits / T

def logits_to_calibrated_probs(logits: torch.Tensor, temp_json_path: Optional[str]) -> torch.Tensor:
    T = load_temperature(temp_json_path)
    return torch.softmax(apply_temperature(logits, T), dim=1)
