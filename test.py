from src.attention import precompute_rotary_emb, apply_rotary_emb
import torch
x = torch.rand((2, 300, 400))
r_emb = precompute_rotary_emb(400, 600)
r_copy = torch.clone(r_emb)
print(r_emb.shape)
res = apply_rotary_emb(x, r_emb)
print(res.shape)