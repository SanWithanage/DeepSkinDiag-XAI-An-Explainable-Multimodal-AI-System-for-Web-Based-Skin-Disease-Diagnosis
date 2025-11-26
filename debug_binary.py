# debug_binary.py — print binary head probs in both orders for one image
import sys, os
from PIL import Image
import torch
import torch.nn.functional as F
import timm
from torchvision import transforms

IMAGE_SIZE = 224
PATH_BIN_CKPT   = "artifacts/checkpoints/binary/best.pt"
PATH_LABELS_BIN = "artifacts/labels_binary.txt"  # must contain 'Healthy' and 'Unhealthy'

def read_lines(p):
    with open(p, "r", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]

img_tf = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

def softmax(x): return F.softmax(x, dim=-1)

def main():
    if len(sys.argv) < 2:
        print("Usage: python debug_binary.py /path/to/image.jpg")
        sys.exit(1)
    img_path = sys.argv[1]
    assert os.path.isfile(img_path), f"not found: {img_path}"

    labels_file = ["Healthy","Unhealthy"]
    if os.path.isfile(PATH_LABELS_BIN):
        labels_file = read_lines(PATH_LABELS_BIN)
    assert "Healthy" in labels_file and "Unhealthy" in labels_file, "labels_binary.txt must list both labels"

    model = timm.create_model("tf_efficientnet_b0", pretrained=False, num_classes=2)
    state = torch.load(PATH_BIN_CKPT, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    model.eval()

    img = Image.open(img_path).convert("RGB")
    x = img_tf(img).unsqueeze(0)
    with torch.inference_mode():
        logits = model(x).squeeze(0)
        probs = softmax(logits).tolist()

    file_order = list(labels_file)          # e.g. ["Healthy","Unhealthy"]
    swapped    = [file_order[1], file_order[0]]

    print("\n[Binary probs USING file order]", file_order)
    for i, p in enumerate(probs):
        print(f"  {file_order[i]:10s} : {p:.4f}")

    print("\n[Binary probs USING SWAPPED order]", swapped)
    for i, p in enumerate(probs):
        print(f"  {swapped[i]:10s} : {p:.4f}")

    a = dict(zip(file_order, probs))
    b = dict(zip(swapped, probs))
    print("\nTakeaway:")
    print(f"  File order says Healthy={a.get('Healthy',0):.4f}  Unhealthy={a.get('Unhealthy',0):.4f}")
    print(f"  Swapped   says Healthy={b.get('Healthy',0):.4f}  Unhealthy={b.get('Unhealthy',0):.4f}")
    print("If 'Swapped' looks correct, enable the 'Swap' checkbox in the UI or set SWAP_BINARY_DEFAULT=True.")

if __name__ == "__main__":
    main()

