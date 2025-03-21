import sys
sys.path.append("..")
from settings import Config
from .sensor_processing import AbstractSensorProcessing
import pathlib
import torch
import torch.nn as nn
from torchvision import transforms, models

class ViTBasicEncoder(nn.Module):
    def __init__(self, exp, device):
        super(ViTBasicEncoder, self).__init__()

        # Configuration
        self.latent_size = exp["latent_size"]
        self.output_size = exp["output_size"]

        # Create ViT model
        if exp["vit_model"] == "vit_b_16":
            from torchvision.models import vit_b_16, ViT_B_16_Weights
            weights = ViT_B_16_Weights.DEFAULT
            self.backbone = vit_b_16(weights=weights)
            vit_dim = 768
        else:
            raise ValueError(f"Unsupported model: {exp['vit_model']}")

        # Replace head with identity
        self.backbone.heads = nn.Identity()

        # Create simple projection and proprioception components
        self.projection = nn.Linear(vit_dim, self.latent_size)
        self.proprioceptor = nn.Linear(self.latent_size, self.output_size)

        # Image preprocessing
        self.resize = transforms.Resize((224, 224))

        # Move to device
        self.to(device)

    def forward(self, x):
        # Resize if needed
        if x.shape[2] != 224:
            x = self.resize(x)

        # Extract features
        features = self.backbone(x)

        # Project to latent space
        latent = self.projection(features)

        # Predict robot position
        output = self.proprioceptor(latent)

        return output

    def encode(self, x):
        # Resize if needed
        if x.shape[2] != 224:
            x = self.resize(x)

        # Extract features
        features = self.backbone(x)

        # Project to latent space
        latent = self.projection(features)

        return latent

class VitSensorProcessing(AbstractSensorProcessing):
    def __init__(self, exp, device="cpu"):
        super().__init__(exp, device)

        # Create encoder
        self.enc = ViTBasicEncoder(exp, device)

        # Load weights if available
        modelfile = pathlib.Path(exp["data_dir"], exp["proprioception_mlp_model_file"])
        if modelfile.exists():
            self.enc.load_state_dict(torch.load(modelfile, map_location=device))

        # Set to eval mode
        self.enc.eval()

    def process(self, sensor_readings):
        with torch.no_grad():
            z = self.enc.encode(sensor_readings)
        return z.cpu().numpy()