# %%
import torch
from gpt import GPTLanguageModel, decode, device  # adjust imports as needed

ckpt = torch.load("ckpt.pth", map_location=device)
model = GPTLanguageModel().to(device)
model.load_state_dict(ckpt["model"])
model.eval()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
# %%
