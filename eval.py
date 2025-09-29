import os
import cv2
import sys
import numpy as np
import tritonclient.grpc as grpcclient
from glog import logger
import time
from torchvision import transforms
from PIL import Image

CONFIG = {
    "hazy": {
        "colormap": {0: "Black", 1: "Grey", 2: "Red", 3: "White"},
        "subfolders": ["White", "Black", "Grey", "Red"],
        "backends": {
            "torch": {"MODEL": "hazy_torch", "INPUT": "INPUT__0", "OUTPUT": "OUTPUT__0"},
            "onnx": {"MODEL": "hazy_onnx", "INPUT": "input", "OUTPUT": "logits"},
            "tensorrt": {"MODEL": "hazy_tensorrt", "INPUT": "input", "OUTPUT": "logits"},
        }
    },
    "night": {
        "colormap": {0: "Black", 1: "Blue", 2: "Grey", 3: "Red", 4: "White", 5: "Yellow"},
        "subfolders": ["Black", "Blue", "Grey", "Red", "White", "Yellow"],
        "backends": {
            "torch": {"MODEL": "night_torch", "INPUT": "INPUT__0", "OUTPUT": "OUTPUT__0"},
            "onnx": {"MODEL": "night_onnx", "INPUT": "input", "OUTPUT": "logits"},
            "tensorrt": {"MODEL": "night_tensorrt", "INPUT": "input", "OUTPUT": "logits"},
        }
    }
}


class Triton:
    def __init__(self, config):
        self.config = config
        self.total_time = 0.0
        try:
            self.triton_server = grpcclient.InferenceServerClient(url=self.config["URL"], 
                                                                  verbose=False)
            logger.info(f"Connected Triton at {self.config['URL']}")
        except Exception as e:
            self.triton_server = None
            logger.error(f"Cannot connect Triton gRPC at {self.config['URL']}: {e}")

    def preprocess(self, img_bgr: np.ndarray) -> np.ndarray:
        if not hasattr(self, "val_transform"):
            self.val_transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

        if img_bgr is None:
            raise ValueError("bad path or unreadable image")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        t = self.val_transform(pil_img).unsqueeze(0)          # (1,3,128,128) float32 CPU
        arr = t.numpy()                                       # NumPy view
        return np.ascontiguousarray(arr, dtype=np.float32)

    def __call__(self, img_path) -> np.ndarray:
        img_bgr = cv2.imread(img_path)
        tensor_nchw = self.preprocess(img_bgr)

        t1 = time.time()
        infer_inputs = [
            grpcclient.InferInput(self.config["INPUT"], tensor_nchw.shape, "FP32")
        ]
        infer_inputs[0].set_data_from_numpy(tensor_nchw)

        infer_outputs = [
            grpcclient.InferRequestedOutput(self.config["OUTPUT"])
        ]

        result = self.triton_server.infer(
            model_name=self.config["MODEL"],
            inputs=infer_inputs,
            outputs=infer_outputs
        )
        logits = result.as_numpy(self.config["OUTPUT"])
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)

        pred_class = probs.argmax(axis=1)[0]
        confidence = probs[0, pred_class]

        t2 = time.time()
        elapsed = t2 - t1
        self.total_time += elapsed
        return pred_class, confidence


def run(triton: Triton, dataset: str, base_dir: str):
    subfolders = CONFIG[dataset]["subfolders"]
    IDX2COLOR = CONFIG[dataset]["colormap"]
    COLOR2IDX = {v: k for k, v in IDX2COLOR.items()}

    image_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
    cnt = 0
    rows = []
    for sub in subfolders:
        gt_idx = COLOR2IDX[sub]  # nhãn thật theo thư mục
        dir_path = os.path.join(base_dir, sub)
        if not os.path.isdir(dir_path):
            logger.warning(f"Skip missing folder: {dir_path}")
            continue

        for name in os.listdir(dir_path):
            if not name.lower().endswith(image_exts):
                continue
            img_path = os.path.join(dir_path, name)
            try:
                pred_idx, conf= triton(img_path)

                rows.append([img_path, gt_idx, pred_idx, conf])
                cnt +=1
                print(cnt)
            except Exception as e:
                logger.error(f"Error on {img_path}: {e}")


    # thống kê nhanh
    total = len(rows)
    correct = sum(1 for r in rows if r[1] == r[2])
    acc = 100.0 * correct / total if total else 0.0

    logger.info(f"Accuracy (folder-as-groundtruth): {acc:.2f}% ({correct}/{total})")
    return acc, cnt

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python main.py <dataset> <backend>")
        print("Example: python main.py hazy/night onnx/tensorrt/torch")
        sys.exit(1)
    
    dataset = sys.argv[1]
    backend = sys.argv[2]
    if dataset not in CONFIG or backend not in CONFIG[dataset]["backends"]:
        raise ValueError(f"Unsupported config: dataset={dataset}, backend={backend}")
    
    backend_cfg = CONFIG[dataset]["backends"][backend]
    triton_cfg = {
        "URL": "localhost:8001",
        "MODEL": backend_cfg["MODEL"],
        "INPUT": backend_cfg["INPUT"],
        "OUTPUT": backend_cfg["OUTPUT"],
    }

    triton = Triton(triton_cfg)
    base_dir = f"/home/vanh/anhnv/color_paper/dataset/{dataset}/test"

    acc, cnt = run(triton, dataset, base_dir)
    print(f"Final Accuracy: {acc:.2f}%")
    print(f"Mean inference time: {(triton.total_time/cnt)*1000:.3f} ms")