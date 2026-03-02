import torch

act = torch.randn(16, 128, 768)
act[0, 0, 0] = 1000.0

_act_scale = act.abs().max().clamp(min=1e-8) / 127.0
_act_q = (act / _act_scale).round().clamp(-127, 127).to(torch.int8)
dequant = _act_q.float() * _act_scale

print("Zero fraction:", (dequant == 0).float().mean().item())

