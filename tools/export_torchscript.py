import torch
from model_ver1 import MobileViTCo

model = MobileViTCo(num_classes=4)
checkpoint = torch.load(
    "/home/vanh/anhnv/color_paper/models/hazy.pth",
    map_location="cpu"
)

model.load_state_dict(checkpoint)
model.eval()

# Dummy input theo đúng shape model của bạn
dummy_input = torch.randn(1, 3, 128, 128)

# Trace hoặc Script
scripted = torch.jit.script(model)

# Save thành TorchScript
torch.jit.save(scripted, "/home/vanh/anhnv/color_paper/models/hazy.pt")
