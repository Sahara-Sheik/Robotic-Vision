"""
custom_vit.py

A custom Vision Transformer (ViT) implementation from scratch for sensor processing.
This implementation does not rely on pre-trained backbones and can be trained
specifically for visual proprioception tasks.
"""
import sys
sys.path.append("..")
from settings import Config

from .sensor_processing import AbstractSensorProcessing

import math
import pathlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


class PatchEmbedding(nn.Module):
    """
    Split image into patches and embed them.

    Args:
        img_size (int): Size of the input image (assumed to be square)
        patch_size (int): Size of each patch (assumed to be square)
        in_channels (int): Number of input channels
        embed_dim (int): Embedding dimension
    """
    def __init__(self, img_size, patch_size, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        # Linear projection to embed patches
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        """
        Args:
            x: Input image tensor of shape [batch_size, in_channels, img_size, img_size]

        Returns:
            Patch embeddings of shape [batch_size, n_patches, embed_dim]
        """
        # Apply projection to get patch embeddings
        x = self.proj(x)  # [B, embed_dim, grid_size, grid_size]

        # Flatten patches into sequence
        x = x.flatten(2)  # [B, embed_dim, n_patches]
        x = x.transpose(1, 2)  # [B, n_patches, embed_dim]

        return x


class Attention(nn.Module):
    """
    Multi-head Self Attention mechanism.

    Args:
        dim (int): Input dimension
        n_heads (int): Number of attention heads
        qkv_bias (bool): Whether to use bias in the QKV projection
        attn_drop (float): Dropout rate for attention weights
        proj_drop (float): Dropout rate for projection
    """
    def __init__(self, dim, n_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        # QKV projection
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, dim]

        Returns:
            Output tensor of shape [batch_size, seq_len, dim]
        """
        batch_size, seq_len, dim = x.shape

        # Calculate QKV
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, n_heads, seq_len, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: [B, n_heads, seq_len, head_dim]

        # Calculate attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, n_heads, seq_len, seq_len]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, dim)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class MLP(nn.Module):
    """
    Multi-Layer Perceptron for transformer.

    Args:
        in_features (int): Input feature dimension
        hidden_features (int): Hidden layer dimension
        out_features (int): Output feature dimension
        drop (float): Dropout rate
    """
    def __init__(self, in_features, hidden_features, out_features, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, in_features]

        Returns:
            Output tensor of shape [batch_size, seq_len, out_features]
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer block with attention and MLP.

    Args:
        dim (int): Input dimension
        n_heads (int): Number of attention heads
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim
        qkv_bias (bool): Whether to use bias in QKV projection
        drop (float): Dropout rate
        attn_drop (float): Attention dropout rate
    """
    def __init__(self, dim, n_heads, mlp_ratio=4.0, qkv_bias=True, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim=dim,
            n_heads=n_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop
        )
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            out_features=dim,
            drop=drop
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, dim]

        Returns:
            Output tensor of shape [batch_size, seq_len, dim]
        """
        # Attention block with residual connection
        x = x + self.attn(self.norm1(x))

        # MLP block with residual connection
        x = x + self.mlp(self.norm2(x))

        return x


class CustomVisionTransformer(nn.Module):
    """
    Custom Vision Transformer implementation from scratch.

    Args:
        img_size (int): Input image size (assumed to be square)
        patch_size (int): Patch size (assumed to be square)
        in_channels (int): Number of input channels
        embed_dim (int): Embedding dimension
        depth (int): Number of transformer blocks
        n_heads (int): Number of attention heads
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim
        qkv_bias (bool): Whether to use bias in QKV projection
        drop_rate (float): Dropout rate
        attn_drop_rate (float): Attention dropout rate
        latent_size (int): Size of the final latent representation
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        depth=12,
        n_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.1,
        attn_drop_rate=0.0,
        latent_size=128
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.n_patches

        # Class token and position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        # Initialize position embeddings
        self._init_weights()

        self.pos_drop = nn.Dropout(p=drop_rate)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                n_heads=n_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Projection to latent space
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, latent_size),
            nn.GELU()
        )

        # Resize transform
        self.resize = transforms.Resize((img_size, img_size), antialias=True)

    def _init_weights(self):
        """Initialize weights for position embeddings"""
        # Initialize cls token
        nn.init.normal_(self.cls_token, std=0.02)

        # Initialize position embeddings
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        """
        Args:
            x: Input image tensor of shape [batch_size, in_channels, height, width]

        Returns:
            Latent representation of shape [batch_size, latent_size]
        """
        # Resize input if needed
        if x.size(2) != x.size(3) or x.size(2) != self.patch_embed.img_size:
            x = self.resize(x)

        # Embed patches
        x = self.patch_embed(x)  # [B, n_patches, embed_dim]
        batch_size = x.shape[0]

        # Prepend class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # [B, 1 + n_patches, embed_dim]

        # Add position embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        # Apply layer norm
        x = self.norm(x)

        # Take class token for representation
        cls_token_final = x[:, 0]  # [B, embed_dim]

        # Project to latent space
        latent = self.projection(cls_token_final)  # [B, latent_size]

        return latent


class CustomVitSensorProcessing(AbstractSensorProcessing):
    """
    Sensor processing using a custom Vision Transformer implementation.

    This class implements the AbstractSensorProcessing interface using a custom
    Vision Transformer model that is built from scratch (no pre-trained backbone).
    """

    def __init__(self, exp, device="cpu"):
        """
        Initialize the custom ViT sensor processing module.

        Args:
            exp (dict): Experiment configuration dictionary
            device (str): Device to run the model on
        """
        super().__init__(exp, device)

        # Log configuration
        print(f"Initializing Custom ViT Sensor Processing:")
        print(f"  Image size: {exp['image_size']}x{exp['image_size']}")
        print(f"  Patch size: {exp['patch_size']}")
        print(f"  Embed dim: {exp['embed_dim']}")
        print(f"  Depth: {exp['depth']}")
        print(f"  Heads: {exp['n_heads']}")
        print(f"  Latent size: {exp['latent_size']}")

        # Create custom ViT model
        self.enc = CustomVisionTransformer(
            img_size=exp["image_size"],
            patch_size=exp["patch_size"],
            in_channels=exp.get("in_channels", 3),
            embed_dim=exp["embed_dim"],
            depth=exp["depth"],
            n_heads=exp["n_heads"],
            mlp_ratio=exp.get("mlp_ratio", 4.0),
            qkv_bias=exp.get("qkv_bias", True),
            drop_rate=exp.get("drop_rate", 0.1),
            attn_drop_rate=exp.get("attn_drop_rate", 0.0),
            latent_size=exp["latent_size"]
        )

        # Move model to device
        self.enc = self.enc.to(device)

        # Load pre-trained weights if available
        modelfile = pathlib.Path(exp["data_dir"], exp["proprioception_mlp_model_file"])
        if modelfile.exists():
            print(f"Loading Custom ViT weights from {modelfile}")
            self.enc.load_state_dict(torch.load(modelfile, map_location=device))
        else:
            print(f"No pre-trained weights found at {modelfile}. Using random initialization.")

        # Set model to evaluation mode
        self.enc.eval()

    def process(self, sensor_readings):
        """
        Process sensor readings (images) to produce embeddings.

        Args:
            sensor_readings: Image tensor prepared into a batch

        Returns:
            Embedding vector as numpy array with dimensions specified in config
        """
        with torch.no_grad():
            z = self.enc(sensor_readings)

        z = torch.squeeze(z)
        return z.cpu().numpy()