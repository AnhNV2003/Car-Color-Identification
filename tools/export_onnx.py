import torch
from model_ver2 import MobileViTCo

def load_checkpoint(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        state = ckpt
    else:
        raise ValueError("Unsupported checkpoint format")

    # Strip 'module.' if present
    new_state = {}
    for k, v in state.items():
        nk = k.replace("module.", "") if k.startswith("module.") else k
        new_state[nk] = v

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing:
        print(f"[Warn] Missing keys ({len(missing)}): {missing[:5]}{' ...' if len(missing)>5 else ''}")
    if unexpected:
        print(f"[Warn] Unexpected keys ({len(unexpected)}): {unexpected[:5]}{' ...' if len(unexpected)>5 else ''}")

def main():
    # Build & load
    model = MobileViTCo(image_size=128,
                        num_classes=4,
                        variant="XXS")
    load_checkpoint(model, "/home/vanh/anhnv/color_paper/models/hazy.pth")
    model.eval()

    # Dummy input (float32, scaled [0,1])
    dummy = torch.rand(1, 3, 128, 128)

    print("Export Time")
    torch.onnx.export(
        model, dummy, "/home/vanh/anhnv/color_paper/models/hazy.onnx",
        input_names=["input"], output_names=["logits"],
        opset_version=16
    )
    print("Done.")

    # (Optional) quick sanity check if onnx/onnxruntime available
    try:
        import onnx
        onnx_model = onnx.load("/home/vanh/anhnv/color_paper/models/hazy.onnx")
        onnx.checker.check_model(onnx_model)
        print("ONNX checker: OK")
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession("/home/vanh/anhnv/color_paper/models/hazy.onnx", providers=["CPUExecutionProvider"])
            out = sess.run(None, {"input": dummy.numpy()})[0]
            print(f"ORT run OK. Output shape: {out.shape}")
        except Exception as e:
            print(f"[Note] onnxruntime not available or run failed: {e}")
    except Exception as e:
        print(f"[Note] onnx checker skipped or failed: {e}")

if __name__ == "__main__":
    main()