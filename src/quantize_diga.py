#!/usr/bin/env python3
import os, sys, argparse
from PIL import Image

REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, REPO_ROOT)

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.ao.quantization import get_default_qconfig
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from src.models.ttda_module import get_deeplab_v2

class CityscapesCalibDataset(Dataset):
    def __init__(self, udata_dir, split="val", transform=None):
        list_txt = os.path.join(udata_dir, "Cityscapes", "advent_list", f"{split}.txt")
        with open(list_txt, 'r') as f:
            rel_paths = [l.strip() for l in f if l.strip()]
        self.files = [os.path.join(udata_dir, "Cityscapes", "leftImg8bit", p) for p in rel_paths]
        self.transform = transform or transforms.ToTensor()
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        return self.transform(img)

def convert_layer5_to_layer6(state_dict):
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith("layer5."):
            new_k = k.replace("layer5.", "layer6.")
            new_state[new_k] = v
        elif k.startswith("module.layer5."):
            new_k = k.replace("module.layer5.", "layer6.")
            new_state[new_k] = v
        else:
            new_state[k] = v
    return new_state

def quantize_model(fp32_ckpt, int8_ckpt, calib_loader, num_classes=19):
    # Load checkpoint
    state = torch.load(fp32_ckpt, map_location="cpu", weights_only=True)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    elif "state_dict" in state:
        state = state["state_dict"]
    
    # Rename layer5 → layer6 for baseline checkpoints
    if any(k.startswith("layer5.") for k in state):
        print("Converting layer5 → layer6 for baseline checkpoint")
        state = convert_layer5_to_layer6(state)

    # Build model
    model = get_deeplab_v2(num_classes=num_classes, multi_level=False)
    model.load_state_dict(state)
    model.eval()

    # Force-register the upsample module by running a dummy forward pass
    with torch.no_grad():
        _ = model(torch.randn(1, 3, 512, 1024))  # This initializes self.upsample

    # Static quantization config
    torch.backends.quantized.engine = "fbgemm"
    model.qconfig = get_default_qconfig("fbgemm")
    
    # Prepare model with FX
    example = torch.randn(1, 3, 512, 1024)
    model_prepared = prepare_fx(model, {}, example_inputs=(example,))

    # Calibration
    with torch.no_grad():
        for i, imgs in enumerate(calib_loader):
            _ = model_prepared(imgs)
            if i >= 20: break  # 20 batches

    # Convert and save
    model_int8 = convert_fx(model_prepared)
    os.makedirs(os.path.dirname(int8_ckpt), exist_ok=True)
    torch.save(model_int8.state_dict(), int8_ckpt)
    print(f"✅ Saved quantized INT8 model to {int8_ckpt}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Quantize DIGA DeepLabV2")
    p.add_argument("--udata_dir", type=str, required=True)
    p.add_argument("--fp32_ckpt", type=str, required=True)
    p.add_argument("--output_int8_ckpt", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--num_classes", type=int, default=19)
    args = p.parse_args()

    # Build calibration dataset
    calib_loader = DataLoader(
        CityscapesCalibDataset(
            args.udata_dir, 
            split="val",
            transform=transforms.ToTensor()
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Run quantization
    quantize_model(
        fp32_ckpt=args.fp32_ckpt,
        int8_ckpt=args.output_int8_ckpt,
        calib_loader=calib_loader,
        num_classes=args.num_classes
    )
