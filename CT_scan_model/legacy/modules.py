import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .configuration import *

class EMA:
    """
    Exponential Moving Average (EMA) class for tracking model parameter updates.
    This is commonly used in deep learning to maintain a smoothed version of model weights.
    """

    def __init__(self, beta):
        """
        Initializes the EMA instance.

        Parameters:
        beta (float): The smoothing factor for EMA, typically close to 1 (e.g., 0.99).
        """
        super().__init__()
        self.beta = beta  # Decay factor for EMA updates
        self.step = 0  # Step counter to track updates

    def update_model_average(self, ma_model, current_model):
        """
        Updates the moving average model parameters using the current model.

        Parameters:
        ma_model: The model maintaining the moving average of parameters.
        current_model: The current model with updated parameters.
        """
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        """
        Computes the new exponential moving average value.

        Parameters:
        old: The previous EMA value.
        new: The new value to incorporate.

        Returns:
        Updated EMA value based on the smoothing factor.
        """
        if old is None:
            return new  # If no previous value, use the new value directly
        return old * self.beta + (1 - self.beta) * new  # EMA update formula

    def step_ema(self, ema_model, model, step_start_ema=2000):
        """
        Updates the EMA model after a certain number of steps.

        Parameters:
        ema_model: The model maintaining the EMA parameters.
        model: The current model with updated parameters.
        step_start_ema (int): The step at which EMA updates should start (default: 2000).
        """
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)  # Sync models before EMA starts
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        """
        Resets the EMA model parameters to match the current model.

        Parameters:
        ema_model: The model maintaining the EMA parameters.
        model: The current model with updated parameters.
        """
        ema_model.load_state_dict(model.state_dict())


class SelfAttention2(nn.Module):
    """
    Implements a self-attention mechanism with 2 attention heads.
    """

    def __init__(self, channels, size):
        super(SelfAttention2, self).__init__()
        self.channels = channels  # Number of input channels
        self.size = size  # Spatial size of the input

        # Multi-head self-attention with 2 heads
        self.mha = nn.MultiheadAttention(channels, 2, batch_first=True)

        # Layer normalization for stable training
        self.ln = nn.LayerNorm([channels])

        # Feed-forward network with GELU activation
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )

    def forward(self, x):
        # Reshape input tensor for attention mechanism
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)

        # Apply layer normalization
        x_ln = self.ln(x)

        # Compute self-attention
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)

        # Add residual connection
        attention_value = attention_value + x

        # Apply feed-forward network and another residual connection
        attention_value = self.ff_self(attention_value) + attention_value

        # Reshape back to original tensor shape
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)


class SelfAttention4(nn.Module):
    """
    Implements a self-attention mechanism with 4 attention heads.
    """

    def __init__(self, channels, size):
        super(SelfAttention4, self).__init__()
        self.channels = channels  # Number of input channels
        self.size = size  # Spatial size of the input

        # Multi-head self-attention with 4 heads
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)

        # Layer normalization for stable training
        self.ln = nn.LayerNorm([channels])

        # Feed-forward network with GELU activation
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )

    def forward(self, x):
        # Reshape input tensor for attention mechanism
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)

        # Apply layer normalization
        x_ln = self.ln(x)

        # Compute self-attention
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)

        # Add residual connection
        attention_value = attention_value + x

        # Apply feed-forward network and another residual connection
        attention_value = self.ff_self(attention_value) + attention_value

        # Reshape back to original tensor shape
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)


class DoubleConv(nn.Module):
    """
    Implements a double convolutional layer with optional residual connection.
    """

    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual  # Flag for residual connection

        # Set mid_channels to out_channels if not specified
        if not mid_channels:
            mid_channels = out_channels

        # Define double convolutional layers with GroupNorm and GELU activation
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels)
        )

    def forward(self, x):
        if self.residual:
            # Apply residual connection with GELU activation
            return F.gelu(x + self.double_conv(x))
        else:
            # Standard double convolution
            return self.double_conv(x)


class Down(nn.Module):
    # DOWNSAMPLING - DIFFUSION
    def __init__(self, in_channels, out_channels, imsize, emb_dim=512):
        super().__init__()
        self.imsize = imsize  # Store the input image size

        # Downsampling block: Max pooling followed by two convolutional layers
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),  # Reduces spatial dimensions by a factor of 2
            DoubleConv(in_channels, in_channels, residual=True),  # First convolutional block with residual connection
            DoubleConv(in_channels, (out_channels - num_conditions))  # Second convolutional block

        )

        # Embedding layer for time step input (t), applying activation and linear transformation
        self.emb_layer = nn.Sequential(
            nn.SiLU(),  # Swish activation function
            nn.Linear(emb_dim, out_channels)  # Linear layer to transform embedding dimension to output channels
        )

        self.fm_label = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, 1 * self.imsize * self.imsize)  # Output reshaped into image size
        )

    def forward(self, x, t, fm):
        # Apply the downsampling convolutions
        x = self.maxpool_conv(x)

        # Compute the embedding for the time step t and expand it spatially
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])

        # Compute the embeddings for categorical inputs and reshape them to match x
        fmemb = self.fm_label(fm)
        fmemb = fmemb.view(x.shape[0], 1, x.shape[-2], x.shape[-1])

        # Concatenate the embeddings along the channel dimension
        # x = torch.cat((x, shpemb, locemb, cremb, skemb, htemb, ftemb, magemb), dim=1)
        x = torch.cat((x, fmemb), dim=1)

        # Return the final output with the time embedding added
        # print(x.size())
        # print(emb.size())
        return x + emb


class Up(nn.Module):
    # DOWNSAMPLING - DENOISING
    def __init__(self, in_channels, out_channels, imsize, emb_dim=512):
        super().__init__()

        # Initialize the input image size
        self.imsize = imsize

        # Define an upsampling layer with a scale factor of 2, using bilinear interpolation
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        # Define a sequence of convolution layers (DoubleConv) for the input channels
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),  # A DoubleConv layer with residual connection
            # DoubleConv(in_channels, out_channels-7, in_channels // 2)  # Another DoubleConv with half of the input channels
            DoubleConv(in_channels, out_channels - num_conditions, in_channels // 2)
            # Another DoubleConv with half of the input channels
        )

        # Define a layer to process embedding data, applying SiLU activation and a linear transformation
        self.emb_layer = nn.Sequential(
            nn.SiLU(),  # Activation function
            nn.Linear(emb_dim, out_channels)  # Linear transformation to match output channels
        )

        # Define separate embedding layers for different labels (shapes, locations, etc.)
        # Each embedding layer maps categorical values to embeddings, and then a linear layer transforms the embedding into a shape matching the output image size (1 * imsize * imsize)

        ##################################################################################################
        # Change conditions here
        ##################################################################################################

        # CONDITIONS

        # FEED MOD OF MICROSTRUCTURE
        self.fm_label = nn.Sequential(
            nn.Linear(emb_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 128),
            nn.SiLU(),
            nn.Linear(128, 1 * self.imsize * self.imsize)
        )

    def forward(self, x, skip_x, t, fm):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)

        ##################################################################################################
        # Change conditions here
        ##################################################################################################

        fmemb = self.fm_label(fm)
        fmemb = fmemb.view(x.shape[0], 1, x.shape[-2], x.shape[-1])

        x = torch.cat((x, fmemb), dim=1)  # Summieren von allen Embedded Input Parametern

        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class UNet(nn.Module):
    def __init__(self, c_in=1, c_out=1, time_dim=256):
        super().__init__()
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 16)
        self.down1 = Down(16, 32)
        self.sa1 = SelfAttention4(32, 128)
        self.down2 = Down(32, 64)
        self.sa2 = SelfAttention4(64, 64)
        self.down3 = Down(64, 128)
        self.sa3 = SelfAttention4(128, 32)
        self.down4 = Down(128, 256)
        self.sa4 = SelfAttention4(256, 16)
        self.down5 = Down(256, 256)
        self.sa5 = SelfAttention4(256, 8)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up5 = Up(512, 128)
        self.as5 = SelfAttention4(128, 16)
        self.up4 = Up(256, 64)
        self.as4 = SelfAttention4(64, 32)
        self.up3 = Up(128, 32)
        self.as3 = SelfAttention4(32, 64)
        self.up2 = Up(64, 16)
        self.as2 = SelfAttention4(16, 128)
        self.up1 = Up(32, 8)
        self.as1 = SelfAttention4(8, 256)
        self.outc = nn.Conv2d(8, c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
                10000
                ** (torch.arange(0, channels, 2).float().to(device) / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        x0 = self.inc(x)
        x1 = self.down1(x0, t)
        # x1 = self.sa1(x1)
        x2 = self.down2(x1, t)
        x2 = self.sa2(x2)
        x3 = self.down3(x2, t)
        x3 = self.sa3(x3)
        x4 = self.down4(x3, t)
        x4 = self.sa4(x4)
        x5 = self.down5(x4, t)
        x5 = self.sa5(x5)

        x5 = self.bot1(x5)
        x5 = self.bot2(x5)
        x5 = self.bot3(x5)

        x = self.up5(x5, x4, t)
        # del x5, x4
        x = self.as5(x)
        x = self.up4(x, x3, t)
        # del x3
        x = self.as4(x)
        x = self.up3(x, x2, t)
        # del x2
        x = self.as3(x)
        x = self.up2(x, x1, t)
        # del x1
        # x = self.as2(x)
        x = self.up1(x, x0, t)
        # del x0
        # x = self.as1(x)
        x = self.outc(x)
        return x



class UNet_conditional(nn.Module):
    def __init__(self, c_in=1, c_out=1, time_dim=512, condition_dim=1):
        super().__init__()
        self.time_dim = time_dim
        self.condition_dim = condition_dim
        self.inc = DoubleConv(c_in, 16)
        self.down1 = Down(16, 32, 256)
        self.sa1 = SelfAttention4(32, 256)
        self.down2 = Down(32, 64, 128)
        self.sa2 = SelfAttention4(64, 128)
        self.down3 = Down(64, 128, 64)
        self.sa3 = SelfAttention2(128, 64)
        self.down4 = Down(128, 256, 32)
        self.sa4 = SelfAttention4(256, 32)
        self.down5 = Down(256, 512, 16)
        self.sa5 = SelfAttention4(512, 16)
        self.down6 = Down(512, 512, 8)
        self.sa6 = SelfAttention4(512, 8)

        self.bot1 = DoubleConv(512, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 512)

        self.up6 = Up(1024, 256, 16)
        self.as6 = SelfAttention4(256, 16)
        self.up5 = Up(512, 128, 32)
        self.as5 = SelfAttention4(128, 32)
        self.up4 = Up(256, 64, 64)
        self.as4 = SelfAttention4(64, 64)
        self.up3 = Up(128, 32, 128)
        self.as3 = SelfAttention2(32, 128)
        self.up2 = Up(64, 16, 256)
        self.as2 = SelfAttention4(16, 256)
        self.up1 = Up(32, 8, 512)
        self.as1 = SelfAttention4(8, 512)
        self.outc = nn.Conv2d(8, c_out, kernel_size=1)

        # Für kontinuierliche Konditionierung - Optional: MLP für Embedding
        self.condition_encoder = nn.Sequential(
            nn.Linear(condition_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 256),
            nn.SiLU(),
            nn.Linear(256, time_dim) #Hier time_dim wegen gleicher embedding größe (für combined Emb)
        )

    def pos_encoding(self, t, channels):
        device = t.device #Added for multi-GPU compatibility

        inv_freq = 1.0 / (
                10000
                ** (torch.arange(0, channels, 2).float().to(device) / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, fm):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        # Kontinuierlichen Parameter verarbeiten
        # Encoding des kontinuierlichen Parameters
        fm = fm.view(-1, 1)  # Batch_size x 1
        fm = self.condition_encoder(fm)

        # (Optional: Kombinieren von Zeit- und Bedingungsembedding)
        # combined_embedding = t + fm

        x0 = self.inc(x)
        x1 = self.down1(x0, t, fm)
        # x1 = self.sa1(x1)
        x2 = self.down2(x1, t, fm)
        # x2 = self.sa2(x2)
        x3 = self.down3(x2, t, fm)
        x3 = self.sa3(x3)
        x4 = self.down4(x3, t, fm)
        x4 = self.sa4(x4)
        x5 = self.down5(x4, t, fm)
        x5 = self.sa5(x5)  
        x6 = self.down6(x5, t, fm)
        x6 = self.sa6(x6)

        x6 = self.bot1(x6)
        x6 = self.bot2(x6)
        x6 = self.bot3(x6)

        x = self.up6(x6, x5, t, fm)
        x = self.as6(x)
        x = self.up5(x, x4, t, fm)
        x = self.as5(x)
        x = self.up4(x, x3, t, fm)
        x = self.as4(x)
        x = self.up3(x, x2, t, fm)
        x = self.as3(x)
        x = self.up2(x, x1, t, fm)
        # x = self.as2(x)
        x = self.up1(x, x0, t, fm)
        # x = self.as1(x)
        output = self.outc(x)

        return output
