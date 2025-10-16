import os
import torch
import torch.nn as nn
from torch.quantization import QuantStub, DeQuantStub
from torch.utils.data import DataLoader
from torchvision import transforms
from src.models.ttda_module import get_deeplab_v2  # or your model loading function
from PIL import Image


# Function to load the pretrained model with weight adjustments
def load_partial_state_dict(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']

    model_dict = model.state_dict()
    new_dict = {}

    for k, v in checkpoint.items():
        if k.startswith("layer5."):
            k = k.replace("layer5", "layer6", 1)

        if k in model_dict and model_dict[k].shape == v.shape:
            new_dict[k] = v
        else:
            print(f"Skipping: {k} due to shape mismatch or not in model.")

    model.load_state_dict({**model_dict, **new_dict})

# Quantization Wrapper Class
class QuantWrapper(nn.Module):
    def __init__(self, model):
        super(QuantWrapper, self).__init__()
        self.model = model
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        out = self.model(x)
        if isinstance(out, list):
            out = [self.dequant(o) for o in out]
        else:
            out = self.dequant(out)
        return out


# Create the dataset for calibration
class CityscapesCalibDataset(torch.utils.data.Dataset):
    def __init__(self, udata_dir, split="val", transform=None):
        list_txt = os.path.join(udata_dir, "Cityscapes", "advent_list", f"{split}.txt")
        with open(list_txt, 'r') as f:
            rel_paths = [l.strip() for l in f if l.strip()]
        self.files = [os.path.join(udata_dir, "Cityscapes", "leftImg8bit","val", p) for p in rel_paths]
        self.transform = transform or transforms.ToTensor()

    def __len__(self): 
        return len(self.files)
    
    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        return self.transform(img)

# Main code for quantization process
model_path = '/u/student/2023/cs23mtech12003/data/models/DA_Seg_models/GTA5/GTA5_baseline.pth'  # Modify with correct path
base_model = get_deeplab_v2()  # This must return a pretrained model architecture
print("called the deeplab v2 model...")
# Load the model with adjusted weights
load_partial_state_dict(base_model, model_path)
base_model.eval()
print("loaded the model...")
# Wrap model for quantization
quant_model = QuantWrapper(base_model)

# Set quantization config
quant_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')

# Prepare for calibration
torch.quantization.prepare(quant_model, inplace=True)

# Build calibration dataset and loader
calib_loader = DataLoader(
    CityscapesCalibDataset(
        udata_dir='/u/student/2023/cs23mtech12003/data',  # Replace with your actual data path
        split="val",
        transform=transforms.ToTensor()
    ),
    batch_size=1,  # Adjust as needed
    shuffle=False,
    num_workers=4,  # Adjust as needed
    pin_memory=True
)

# Calibrate with the dataset
with torch.no_grad():
    for inputs in calib_loader:
        quant_model(inputs)

# Convert to quantized model
torch.quantization.convert(quant_model, inplace=True)

# Save quantized model
torch.save(quant_model.state_dict(), 'deeplabv2_quantized.pth')

print("Quantized model saved successfully.")