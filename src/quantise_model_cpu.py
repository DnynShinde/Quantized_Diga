import os
import torch
import torch.nn as nn
from torch.quantization import QuantStub, DeQuantStub
from torch.utils.data import DataLoader
from torchvision import transforms
from src.models.ttda_module import get_deeplab_v2  # Imports the function to load the DeepLab v2 model
from PIL import Image


# Function to load pretrained model weights with potential layer name adjustments
def load_partial_state_dict(model, checkpoint_path):
    """Loads weights from a checkpoint into the model, handling potential layer name differences."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')  # Load checkpoint, ensuring CPU compatibility
    if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']  # Extract state dictionary if present
    model_dict = model.state_dict()  # Get the current state dictionary of the model
    new_dict = {}  # Initialize a dictionary for the weights to be loaded

    for k, v in checkpoint.items():  # Iterate through layers and weights in the checkpoint
        if k.startswith("layer5."):
            k = k.replace("layer5", "layer6", 1)  # Adjust layer name if it starts with "layer5"
        if k in model_dict and model_dict[k].shape == v.shape:
            new_dict[k] = v  # Add weight to the new dictionary if layer exists and shapes match
        else:
            print(f"Skipping: {k} due to shape mismatch or not in model.")

    model.load_state_dict({**model_dict, **new_dict})  # Load the compatible weights into the model


# Wrapper class to prepare a model for quantization
class QuantWrapper(nn.Module):
    """Wraps a model and adds quantization and dequantization stubs."""
    def __init__(self, model):
        super(QuantWrapper, self).__init__()
        self.model = model  # Store the original model
        self.quant = QuantStub()  # Marks where quantization of the input will begin
        self.dequant = DeQuantStub()  # Marks where dequantization of the output will occur

    def forward(self, x):
        x = self.quant(x)  # Quantize the input tensor
        out = self.model(x)  # Pass the quantized input through the model
        if isinstance(out, list):
            out = [self.dequant(o) for o in out]  # Dequantize each output tensor if the output is a list
        else:
            out = self.dequant(out)  # Dequantize the single output tensor
        return out


# Dataset for calibration
class CityscapesCalibDataset(torch.utils.data.Dataset):
    """Loads Cityscapes images for quantization calibration."""
    def __init__(self, udata_dir, split="val", transform=None):
        list_txt = os.path.join(udata_dir, "Cityscapes", "advent_list", f"{split}.txt")  # Path to the file listing image paths
        with open(list_txt, 'r') as f:
            rel_paths = [l.strip() for l in f if l.strip()]  # Read and clean relative image paths
        self.files = [os.path.join(udata_dir, "Cityscapes", "leftImg8bit","val", p) for p in rel_paths]  # Create full image paths
        self.transform = transform or transforms.ToTensor()  # Use provided transform or default to ToTensor

    def __len__(self):
        return len(self.files)  # Return the number of images in the dataset

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")  # Open image and convert to RGB
        return self.transform(img)  # Apply transformation to the image


# --- Main quantization process ---
# Define the path to the pretrained model weights
model_path = '/u/student/2023/cs23mtech12003/data/models/DA_Seg_models/GTA5/GTA5_baseline.pth'
# Load the base DeepLab v2 model architecture
base_model = get_deeplab_v2()
print("called the deeplab v2 model...")
# Load pretrained weights into the base model, handling potential layer name mismatches
load_partial_state_dict(base_model, model_path)
# Set the model to evaluation mode (important for quantization)
base_model.eval()
print("loaded the model...")
# Wrap the base model with the QuantWrapper
quant_model = QuantWrapper(base_model)

# Set the quantization configuration for CPU inference using the FBGEMM engine
quant_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')

# Prepare the model for calibration. This step inserts observer modules
# that will record activation statistics during the calibration phase.
torch.quantization.prepare(quant_model, inplace=True)

# Create a DataLoader for the calibration dataset
calib_loader = DataLoader(
    CityscapesCalibDataset(
        udata_dir='/u/student/2023/cs23mtech12003/data',
        split="val",
        transform=transforms.ToTensor()
    ),
    batch_size=1,  # Use a batch size of 1 for calibration (can be adjusted)
    shuffle=False,  # Shuffling is not needed for calibration
    num_workers=4,  # Number of data loading workers
    pin_memory=True   # Improves data transfer to GPU if applicable (though we are quantizing for CPU here)
)

# Calibration loop: Feed data through the prepared model to collect statistics
# These statistics will be used to determine the quantization parameters.
with torch.no_grad():  # Disable gradient calculations during calibration
    for inputs in calib_loader:
        quant_model(inputs)  # Perform a forward pass to trigger the observers

# Convert the model to its quantized form. This replaces floating-point operations
# with their fixed-point counterparts based on the collected statistics.
torch.quantization.convert(quant_model, inplace=True)

# Save the state dictionary of the fully quantized model
torch.save(quant_model.state_dict(), '/u/student/2023/cs23mtech12003/data/models/DA_Seg_models/GTA5Quantisedatadeeplabv2_quantized_cpu.pth')

print("Quantized model saved successfully.")