import torch
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image

def get_gradcam_overlay(pil_img, model_path="artifacts/checkpoints/multiclass/best.pt", target_layer_name="conv_head"):
    import timm
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=26)
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])
    img_tensor = preprocess(pil_img).unsqueeze(0)

    grads, fmap = {}, {}
    def save_fmap(name):
        def hook(module, input, output): fmap[name] = output
        return hook
    def save_grad(name):
        def hook(module, grad_in, grad_out): grads[name] = grad_out[0]
        return hook

    layer = dict([*model.named_modules()])[target_layer_name]
    layer.register_forward_hook(save_fmap(target_layer_name))
    layer.register_backward_hook(save_grad(target_layer_name))

    out = model(img_tensor)
    pred_class = out.argmax(dim=1).item()
    out[:, pred_class].backward()

    grad = grads[target_layer_name].mean(dim=(2,3), keepdim=True)
    cam = (fmap[target_layer_name] * grad).sum(dim=1).squeeze()
    cam = torch.relu(cam)
    cam = cam - cam.min()
    cam = cam / cam.max()
    cam = cam.detach().numpy()

    cam = cv2.resize(cam, pil_img.size)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = np.float32(heatmap) * 0.4 + np.float32(pil_img)
    overlay = np.uint8(overlay / overlay.max() * 255)
    return Image.fromarray(overlay)
