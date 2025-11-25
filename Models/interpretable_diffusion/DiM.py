import torch
import torch.nn as nn
import numpy as np
import math
from Models.interpretable_diffusion.SMamba import StructureAwareSSM, SelectSSM
from functools import partial
from itertools import repeat
import collections.abc


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)
    
class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks

    NOTE: When use_conv=True, expects 2D NCHW tensors, otherwise N*C expected.
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class TimestepEmbedder(nn.Module):

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LearnablePositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(
            torch.empty(1, max_len, d_model)
        )  # requires_grad automatically set to True
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [batch size, sequence length, embed dim]
            output: [batch size, sequence length, embed dim]
        """
        # print(x.shape)
        x = x + self.pe
        return self.dropout(x)


class DiMBlock(nn.Module):

    def __init__(self, hidden_size, mlp_ratio=4.0, dstate=2, dconv=2, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        self.attn1 = SelectSSM(d_model=hidden_size,d_state=dstate,d_conv=dconv,)
        self.attn2 = SelectSSM(d_model=hidden_size,d_state=dstate,d_conv=dconv,)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        '''
        x = x + gate_msa.unsqueeze(1) * (self.attn1(modulate(self.norm1(x), shift_msa, scale_msa))+self.attn2(modulate(self.norm1(x), shift_msa, scale_msa).flip(dims=[1])).flip(dims=[1]))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x
        '''
        x1, x0, dt, A, B, C, bias = self.attn1(modulate(self.norm1(x), shift_msa, scale_msa))

        x = x + gate_msa.unsqueeze(1) * (x1+self.attn2(modulate(self.norm1(x), shift_msa, scale_msa).flip(dims=[1]))[0].flip(dims=[1]))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x, x0, dt, A, B, C, bias
    
class DiSBlock(nn.Module):

    def __init__(self, hidden_size, mlp_ratio=4.0, dstate=2, dconv=2, conv_num=3, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        self.attn1 = StructureAwareSSM(d_model=hidden_size,d_state=dstate,d_conv=dconv,conv_num=conv_num,)
        self.attn2 = StructureAwareSSM(d_model=hidden_size,d_state=dstate,d_conv=dconv,conv_num=conv_num,)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )

        x1, x0, dt, A, B, C, bias = self.attn1(modulate(self.norm1(x), shift_msa, scale_msa))

        x = x + gate_msa.unsqueeze(1) * (x1+self.attn2(modulate(self.norm1(x), shift_msa, scale_msa).flip(dims=[1]))[0].flip(dims=[1]))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x, x0, dt, A, B, C, bias


class MambaEncoderBlock(nn.Module):

    def __init__(self, hidden_size, mlp_ratio=4.0, dstate=2, dconv=2, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn1 = SelectSSM(d_model=hidden_size,d_state=dstate,d_conv=dconv,)
        self.attn2 = SelectSSM(d_model=hidden_size,d_state=dstate,d_conv=dconv,)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )

    def forward(self, x):
        x = x + self.attn1(self.norm1(x))[0] + self.attn2(self.norm1(x).flip(dims=[1]))[0].flip(dims=[1])
        x = x + self.mlp(self.norm2(x))
        return x


class Encoder(nn.Module):
    def __init__(self, hidden_size=512, num_heads=8, n_layers=3, mlp_ratio=4.0, d_state=2, d_conv=2):
        super().__init__()
        
        self.encoder_blocks = nn.Sequential(
            *[
                MambaEncoderBlock(
                    hidden_size=hidden_size, mlp_ratio=mlp_ratio, dstate=d_state, dconv=d_conv
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x):
        for index in range(len(self.encoder_blocks)):
            x = self.encoder_blocks[index](x)
        return x


class Decoder_M(nn.Module):

    def __init__(self, hidden_size=512, num_heads=8, n_layers=3, mlp_ratio=4.0, d_state=2, d_conv=2):
        super().__init__()
        
        self.encoder_blocks = nn.Sequential(
            *[
                DiMBlock(
                    hidden_size=hidden_size, mlp_ratio=mlp_ratio, dstate=d_state, dconv=d_conv
                )
                for _ in range(n_layers)
            ]
        )
        self.diffusion_step_emb = TimestepEmbedder(hidden_size)

    def forward(self, x, t):
        identity = x
        toreturn = torch.zeros_like(x)
        c = self.diffusion_step_emb(t)
        x_set = []
        dt_set = []
        A_set = []
        B_set = []
        C_set = []
        bias_set = []
        for index in range(len(self.encoder_blocks)):
            x, x0, dt, A, B, C, bias = self.encoder_blocks[index](x, c)
            toreturn += x
            x += identity
            identity = x

            x_set.append(x0)
            dt_set.append(dt)
            A_set.append(A)
            B_set.append(B)
            C_set.append(C)
            bias_set.append(bias)

        return toreturn, x_set, dt_set, A_set, B_set, C_set, bias_set


class Decoder_S(nn.Module):

    def __init__(self, hidden_size=512, num_heads=8, n_layers=3, mlp_ratio=4.0, d_state=2, d_conv=2, conv_num=3):
        super().__init__()
        
        self.encoder_blocks = nn.Sequential(
            *[
                DiSBlock(
                    hidden_size=hidden_size, mlp_ratio=mlp_ratio, dstate=d_state, dconv=d_conv, conv_num=conv_num
                )
                for _ in range(n_layers)
            ]
        )
        self.diffusion_step_emb = TimestepEmbedder(hidden_size)

    def forward(self, x, t):
        identity = x
        toreturn = torch.zeros_like(x)
        c = self.diffusion_step_emb(t)
        x_set = []
        dt_set = []
        A_set = []
        B_set = []
        C_set = []
        bias_set = []
        for index in range(len(self.encoder_blocks)):

            x, x0, dt, A, B, C, bias = self.encoder_blocks[index](x, c)
            toreturn += x
            x += identity
            identity = x

            x_set.append(x0)
            dt_set.append(dt)
            A_set.append(A)
            B_set.append(B)
            C_set.append(C)
            bias_set.append(bias)

        return toreturn


class TimeSeries2EmbLinear(nn.Module):
    """
    Encode time series data alone with selected dimension.
    """

    def __init__(
        self,
        hidden_size=512,
        feature_last=True,
        shape=(24, 6),
        dim2emb="time",
        dropout=0,
    ):
        super().__init__()
        assert dim2emb in ["time", "feature"], "Please indicate which dim to emb"

        if feature_last:
            sequence_length, feature_size = shape
        else:
            feature_size, sequence_length = shape

        self.feature_last = feature_last
        self.dim2emb = dim2emb
        self.pos_emb = LearnablePositionalEncoding(
            d_model=hidden_size, max_len=sequence_length
        )
        if dim2emb == "time":
            self.processing = nn.Sequential(
                nn.Linear(feature_size, hidden_size), nn.Dropout(dropout)
            )
        else:
            self.processing = nn.Sequential(
                nn.Linear(sequence_length, hidden_size), nn.Dropout(dropout)
            )

    def forward(self, x):
        if not self.feature_last:
            x = x.permute(0, 2, 1)

        if self.dim2emb == "time":
            x = self.processing(x)
            return self.pos_emb(x)
        return self.processing(x.permute(0, 2, 1))


class DiM(nn.Module):
    def __init__(
        self,
        hidden_size=512,
        num_heads=4,
        n_encoder=2,
        n_decoder=2,
        feature_last=True,
        mlp_ratio=4.0,
        dropout=0,
        input_shape=(24, 6),
        d_state=2,
        d_conv=2,
        conv_num=3,
        trans_mx=None,
    ):
        super().__init__()
        self.time2emb = TimeSeries2EmbLinear(
            hidden_size=hidden_size,
            feature_last=feature_last,
            shape=input_shape,
            dim2emb="time",
            dropout=dropout,
        )
        self.feature2emb = TimeSeries2EmbLinear(
            hidden_size=hidden_size,
            feature_last=feature_last,
            shape=input_shape,
            dim2emb="feature",
            dropout=dropout,
        )

        self.time_encoder = Encoder(
            hidden_size=hidden_size,
            num_heads=num_heads,
            n_layers=n_encoder,
            mlp_ratio=mlp_ratio,
            d_state=d_state,
            d_conv=d_conv,
        )
        self.feature_encoder = Encoder(
            hidden_size=hidden_size,
            num_heads=num_heads,
            n_layers=n_encoder,
            mlp_ratio=mlp_ratio,
            d_state=d_state,
            d_conv=d_conv,
        )

        self.time_blocks = Decoder_S(
            hidden_size=hidden_size, num_heads=num_heads, n_layers=n_decoder, d_state=d_state, d_conv=d_conv, conv_num=conv_num
        )
        self.feature_blocks = Decoder_M(
            hidden_size=hidden_size, num_heads=num_heads, n_layers=n_decoder, d_state=d_state, d_conv=d_conv,
        )

        self.fc_time = nn.Linear(hidden_size, input_shape[1])
        self.fc_feature = nn.Linear(hidden_size, input_shape[0])

        self.trans_mx = trans_mx

    def forward(self, x, t):

        x_time = self.time2emb(x)
        x_time = self.time_encoder(x_time)
        x_time = self.time_blocks(x_time, t)
        x_time = self.fc_time(x_time)

        if self.trans_mx is None:
            x_feature = self.feature2emb(x)
        else:
            x_feature = self.feature2emb(torch.matmul(self.trans_mx, x.permute(0, 2, 1)).permute(0, 2, 1))

        x_feature = self.feature_encoder(x_feature)
        x_feature, x0, dt, A, B, C, bias = self.feature_blocks(x_feature, t)
        x_feature = self.fc_feature(x_feature)

        if self.trans_mx is None:
            return x_feature.permute(0, 2, 1) + x_time
        else:
            x_feature = torch.matmul(torch.linalg.inv(self.trans_mx), x_feature)
            return x_feature.permute(0, 2, 1) + x_time
