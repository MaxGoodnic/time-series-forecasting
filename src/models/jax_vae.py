"""
JAX VAE models for time series forecasting.

This module contains Variational Autoencoder implementations using JAX/Flax
that can be adapted for time series forecasting tasks.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass


class SpatialNorm(nn.Module):
    """Spatial normalization layer for JAX."""
    f_channels: int
    zq_channels: Optional[int] = None
    norm_layer: nn.Module = nn.GroupNorm
    freeze_norm_layer: bool = False
    add_conv: bool = False
    norm_layer_params: Dict[str, Any] = None

    def setup(self):
        if self.norm_layer_params is None:
            self.norm_layer_params = {"num_groups": 32, "eps": 1e-6, "use_bias": True}
        
        self.norm = self.norm_layer(**self.norm_layer_params)(self.f_channels)
        
        if self.zq_channels is not None:
            if self.add_conv:
                self.conv = nn.Conv(
                    features=self.zq_channels,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding='SAME',
                    use_bias=True
                )
            
            self.conv_y = nn.Conv(
                features=self.f_channels,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding='SAME',
                use_bias=True
            )
            self.conv_b = nn.Conv(
                features=self.f_channels,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding='SAME',
                use_bias=True
            )

    def __call__(self, f, zq=None):
        norm_f = self.norm(f)
        
        if zq is not None:
            f_size = f.shape[-2:]
            zq_resized = jax.image.resize(zq, (*zq.shape[:-2], *f_size), method='nearest')
            
            if self.add_conv:
                zq_resized = self.conv(zq_resized)
            
            norm_f = norm_f * self.conv_y(zq_resized) + self.conv_b(zq_resized)
        
        return norm_f


def Normalize(in_channels, zq_ch=None, add_conv=None):
    return SpatialNorm(
        f_channels=in_channels,
        zq_channels=zq_ch,
        norm_layer=nn.GroupNorm,
        freeze_norm_layer=False,
        add_conv=add_conv,
        norm_layer_params={"num_groups": 32, "eps": 1e-6, "use_bias": True}
    )


class Upsample(nn.Module):
    """Upsample layer for JAX."""
    in_channels: int
    with_conv: bool

    def setup(self):
        if self.with_conv:
            self.conv = nn.Conv(
                features=self.in_channels,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding='SAME',
                use_bias=True
            )

    def __call__(self, x):
        # Upsample by factor of 2
        x_up = jax.image.resize(x, (x.shape[0], x.shape[1] * 2, x.shape[2] * 2, x.shape[3]), method='nearest')
        
        if self.with_conv:
            x_up = self.conv(x_up)
        
        return x_up


class Downsample(nn.Module):
    """Downsample layer for JAX."""
    in_channels: int
    with_conv: bool

    def setup(self):
        if self.with_conv:
            self.conv = nn.Conv(
                features=self.in_channels,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding='VALID',
                use_bias=True
            )

    def __call__(self, x):
        if self.with_conv:
            # Pad for valid convolution
            x_padded = jnp.pad(x, ((0, 0), (0, 1), (0, 1), (0, 0)), mode='constant')
            return self.conv(x_padded)
        else:
            # Average pooling
            return nn.avg_pool(x, window_shape=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='VALID')


class ResnetBlock(nn.Module):
    """ResNet block for JAX."""
    in_channels: int
    out_channels: Optional[int] = None
    conv_shortcut: bool = False
    dropout: float = 0.0
    temb_channels: int = 512
    zq_ch: Optional[int] = None
    add_conv: bool = False

    def setup(self):
        self.out_channels = self.in_channels if self.out_channels is None else self.out_channels
        self.use_conv_shortcut = self.conv_shortcut

        self.norm1 = Normalize(self.in_channels, self.zq_ch, self.add_conv)
        self.conv1 = nn.Conv(
            features=self.out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='SAME',
            use_bias=True
        )
        
        if self.temb_channels > 0:
            self.temb_proj = nn.Dense(features=self.out_channels)
        
        self.norm2 = Normalize(self.out_channels, self.zq_ch, self.add_conv)
        self.dropout_layer = nn.Dropout(rate=self.dropout, deterministic=False)
        self.conv2 = nn.Conv(
            features=self.out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='SAME',
            use_bias=True
        )
        
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut_layer = nn.Conv(
                    features=self.out_channels,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding='SAME',
                    use_bias=True
                )
            else:
                self.nin_shortcut = nn.Conv(
                    features=self.out_channels,
                    kernel_size=(1, 1),
                    strides=(1, 1),
                    padding='SAME',
                    use_bias=True
                )

    def __call__(self, x, temb=None, zq=None, training=True):
        h = x
        h = self.norm1(h, zq)
        h = nn.swish(h)  # Using swish instead of x * sigmoid(x) for better JAX compatibility
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nn.swish(temb))[:, :, None, None]

        h = self.norm2(h, zq)
        h = nn.swish(h)
        h = self.dropout_layer(h, deterministic=not training)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut_layer(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class TimeSeriesEncoder(nn.Module):
    """VAE Encoder adapted for time series data."""
    ch: int
    out_ch: int
    in_channels: int
    sequence_length: int
    z_channels: int
    ch_mult: Tuple[int, ...] = (1, 2, 4, 8)
    num_res_blocks: int = 2
    dropout: float = 0.0
    resamp_with_conv: bool = True
    double_z: bool = True

    def setup(self):
        self.num_resolutions = len(self.ch_mult)
        
        # Input convolution - adapt for 1D time series
        self.conv_in = nn.Conv(
            features=self.ch,
            kernel_size=(3,),  # 1D convolution for time series
            strides=(1,),
            padding='SAME',
            use_bias=True
        )

        # Downsampling blocks
        curr_res = self.sequence_length
        in_ch_mult = (1,) + tuple(self.ch_mult)
        
        self.down_blocks = []
        for i_level in range(self.num_resolutions):
            block = []
            block_in = self.ch * in_ch_mult[i_level]
            block_out = self.ch * self.ch_mult[i_level]
            
            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=0,
                        dropout=self.dropout
                    )
                )
                block_in = block_out
            
            down_block = {
                'block': block,
            }
            
            if i_level != self.num_resolutions - 1:
                down_block['downsample'] = DownsampleTimeSeries(block_in, self.resamp_with_conv)
                curr_res = curr_res // 2
            
            self.down_blocks.append(down_block)

        # Middle blocks
        self.mid_block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=0,
            dropout=self.dropout
        )
        self.mid_block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=0,
            dropout=self.dropout
        )

        # Output normalization and convolution
        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv(
            features=2 * self.z_channels if self.double_z else self.z_channels,
            kernel_size=(3,),
            strides=(1,),
            padding='SAME',
            use_bias=True
        )

    def __call__(self, x, training=True):
        temb = None

        # Input convolution
        h = self.conv_in(x)

        # Downsampling
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down_blocks[i_level]['block'][i_block](h, temb, training=training)
            
            if i_level != self.num_resolutions - 1:
                h = self.down_blocks[i_level]['downsample'](h)

        # Middle blocks
        h = self.mid_block_1(h, temb, training=training)
        h = self.mid_block_2(h, temb, training=training)

        # Output
        h = self.norm_out(h)
        h = nn.swish(h)
        h = self.conv_out(h)

        return h


class DownsampleTimeSeries(nn.Module):
    """Downsample layer adapted for 1D time series."""
    in_channels: int
    with_conv: bool

    def setup(self):
        if self.with_conv:
            self.conv = nn.Conv(
                features=self.in_channels,
                kernel_size=(3,),
                strides=(2,),
                padding='VALID',
                use_bias=True
            )

    def __call__(self, x):
        if self.with_conv:
            # Pad for valid convolution
            x_padded = jnp.pad(x, ((0, 0), (0, 1), (0, 0)), mode='constant')
            return self.conv(x_padded)
        else:
            # Average pooling over time dimension
            return nn.avg_pool(x, window_shape=(1, 2, 1), strides=(1, 2, 1), padding='VALID')


class TimeSeriesDecoder(nn.Module):
    """VAE Decoder adapted for time series data."""
    ch: int
    out_ch: int
    in_channels: int
    sequence_length: int
    z_channels: int
    ch_mult: Tuple[int, ...] = (1, 2, 4, 8)
    num_res_blocks: int = 2
    dropout: float = 0.0
    resamp_with_conv: bool = True
    give_pre_end: bool = False

    def setup(self):
        self.num_resolutions = len(self.ch_mult)
        
        # Compute dimensions
        in_ch_mult = (1,) + tuple(self.ch_mult)
        block_in = self.ch * self.ch_mult[self.num_resolutions - 1]

        # Input convolution
        self.conv_in = nn.Conv(
            features=block_in,
            kernel_size=(3,),
            strides=(1,),
            padding='SAME',
            use_bias=True
        )

        # Middle blocks
        self.mid_block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=0,
            dropout=self.dropout
        )
        self.mid_block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=0,
            dropout=self.dropout
        )

        # Upsampling blocks
        self.up_blocks = []
        for i_level in reversed(range(self.num_resolutions)):
            block = []
            block_out = self.ch * self.ch_mult[i_level]
            
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=0,
                        dropout=self.dropout
                    )
                )
                block_in = block_out
            
            up_block = {
                'block': block,
            }
            
            if i_level != 0:
                up_block['upsample'] = UpsampleTimeSeries(block_in, self.resamp_with_conv)
            
            self.up_blocks.insert(0, up_block)  # Prepend for consistent order

        # Output normalization and convolution
        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv(
            features=self.out_ch,
            kernel_size=(3,),
            strides=(1,),
            padding='SAME',
            use_bias=True
        )

    def __call__(self, z, training=True):
        # Input convolution
        h = self.conv_in(z)

        # Middle blocks
        h = self.mid_block_1(h, None, None, training=training)
        h = self.mid_block_2(h, None, None, training=training)

        # Upsampling
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up_blocks[i_level]['block'][i_block](h, None, None, training=training)
            
            if i_level != 0:
                h = self.up_blocks[i_level]['upsample'](h)

        # Output
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nn.swish(h)
        h = self.conv_out(h)
        return h


class UpsampleTimeSeries(nn.Module):
    """Upsample layer adapted for 1D time series."""
    in_channels: int
    with_conv: bool

    def setup(self):
        if self.with_conv:
            self.conv = nn.Conv(
                features=self.in_channels,
                kernel_size=(3,),
                strides=(1,),
                padding='SAME',
                use_bias=True
            )

    def __call__(self, x):
        # Upsample by factor of 2 in time dimension
        x_up = jax.image.resize(x, (x.shape[0], x.shape[1] * 2, x.shape[2]), method='nearest')
        
        if self.with_conv:
            x_up = self.conv(x_up)
        
        return x_up


class TimeSeriesVAE(nn.Module):
    """Variational Autoencoder for time series forecasting."""
    generator_params: Dict[str, Any]

    def setup(self):
        z_channels = self.generator_params["z_channels"]
        
        self.encoder = TimeSeriesEncoder(**self.generator_params)
        self.quant_conv = nn.Conv(
            features=z_channels,
            kernel_size=(1,),
            strides=(1,),
            padding='SAME',
            use_bias=True
        )
        self.post_quant_conv = nn.Conv(
            features=z_channels,
            kernel_size=(1,),
            strides=(1,),
            padding='SAME',
            use_bias=True
        )
        
        # Create decoder params
        decoder_params = {
            "ch": self.generator_params["ch"],
            "out_ch": self.generator_params["out_ch"],
            "in_channels": self.generator_params["in_channels"],
            "sequence_length": self.generator_params["sequence_length"],
            "z_channels": self.generator_params["z_channels"],
            "ch_mult": self.generator_params["ch_mult"],
            "num_res_blocks": self.generator_params["num_res_blocks"],
            "dropout": self.generator_params["dropout"],
            "resamp_with_conv": self.generator_params["resamp_with_conv"],
            "give_pre_end": self.generator_params.get("give_pre_end", False)
        }
        self.decoder = TimeSeriesDecoder(**decoder_params)

    def __call__(self, x, training=True):
        """Forward pass through the complete VAE model."""
        # Encode
        h = self.encoder(x, training=training)
        h = self.quant_conv(h)
        
        # Decode
        decoder_input = self.post_quant_conv(h)
        decoded = self.decoder(decoder_input, training=training)
        return decoded

    def encode(self, x, training=True):
        """Encode input to latent space."""
        h = self.encoder(x, training=training)
        h = self.quant_conv(h)
        return h

    def decode(self, quant, training=True):
        """Decode latent to reconstruction."""
        decoder_input = self.post_quant_conv(quant)
        decoded = self.decoder(decoder_input, training=training)
        return decoded


def create_timeseries_vae(
    sequence_length: int,
    num_features: int,
    z_channels: int = 8,
    ch: int = 64,
    ch_mult: Tuple[int, ...] = (1, 2, 4),
    num_res_blocks: int = 2,
    dropout: float = 0.1
) -> TimeSeriesVAE:
    """
    Create a TimeSeriesVAE model with specified parameters.
    
    Args:
        sequence_length: Length of input sequences
        num_features: Number of features in the time series
        z_channels: Number of latent channels
        ch: Base number of channels
        ch_mult: Channel multiplier for each level
        num_res_blocks: Number of residual blocks
        dropout: Dropout rate
        
    Returns:
        TimeSeriesVAE model instance
    """
    generator_params = {
        "ch": ch,
        "out_ch": num_features,
        "in_channels": num_features,
        "sequence_length": sequence_length,
        "z_channels": z_channels,
        "ch_mult": ch_mult,
        "num_res_blocks": num_res_blocks,
        "dropout": dropout,
        "resamp_with_conv": True,
        "double_z": True
    }
    
    return TimeSeriesVAE(generator_params)


def initialize_vae_params(model, sequence_length: int, num_features: int, key: jax.random.PRNGKey):
    """
    Initialize VAE model parameters.
    
    Args:
        model: TimeSeriesVAE model instance
        sequence_length: Length of input sequences
        num_features: Number of features
        key: Random key for initialization
        
    Returns:
        Initialized model parameters
    """
    dummy_input = jnp.ones((1, sequence_length, num_features))
    params = model.init(key, dummy_input, training=False)
    return params
