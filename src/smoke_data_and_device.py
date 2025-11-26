import torch
from torchvision import datasets, transforms

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.set_default_device(device)
    print("Device:", device)

    tf = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor()
    ])

    bin_train = datasets.ImageFolder("data/binary/train", transform=tf)
    mc_train  = datasets.ImageFolder("data/multiclass/train", transform=tf)

    print("Binary classes:", bin_train.classes)
    print("Multiclass classes (26):", len(mc_train.classes))

    # num_workers=0 avoids multiprocessing (safe on macOS/Python 3.13)
    bin_loader = torch.utils.data.DataLoader(bin_train, batch_size=16, shuffle=True, num_workers=0)
    mc_loader  = torch.utils.data.DataLoader(mc_train,  batch_size=16, shuffle=True, num_workers=0)

    xb, yb = next(iter(bin_loader))
    xm, ym = next(iter(mc_loader))
    print("Binary batch:", xb.shape, yb.shape, xb.device)
    print("Multiclass batch:", xm.shape, ym.shape, xm.device)

    import torch.nn as nn
    model = nn.Sequential(
        nn.Conv2d(3,16,3,1,1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        nn.Linear(16,2)
    ).to(device)

    with torch.inference_mode():
        out = model(xb.to(device))
    print("Forward OK:", out.shape)

if __name__ == "__main__":
    main()
