import torch
import torch.nn as nn
import torch.nn.functional as nnf


from kmja import KMJA
from mutual_model_kanv import MKANv


class SpatialTransformer(nn.Module):
    def __init__(self, size, mode="bilinear"):
        super().__init__()
        self.mode = mode
        vectors = [torch.arange(size_value) for size_value in size]
        grid = torch.stack(torch.meshgrid(vectors, indexing="ij")).unsqueeze(0).float()
        self.register_buffer("grid", grid)

    def forward(self, src, flow):
        new_locs = self.grid + flow
        shape = flow.shape[2:]
        for dimension, size_value in enumerate(shape):
            new_locs[:, dimension] = 2 * (
                new_locs[:, dimension] / (size_value - 1) - 0.5
            )
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)[..., [2, 1, 0]]
        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        alpha=0.1,
    ):
        super().__init__()
        self.main = nn.Conv3d(
            in_channels, out_channels, kernel_size, stride, padding
        )
        self.activation = nn.LeakyReLU(alpha)

    def forward(self, x):
        return self.activation(self.main(x))


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channel):
        super().__init__()
        self.Conv1 = ConvBlock(in_channels, out_channel)
        self.Conv2 = nn.Conv3d(out_channel, 3, 3, padding=1)
        nn.init.normal_(self.Conv2.weight, mean=0.0, std=1e-5)
        if self.Conv2.bias is not None:
            nn.init.zeros_(self.Conv2.bias)

    def forward(self, x, y=None):
        if y is not None:
            x = torch.cat([x, y], dim=1)
        x = self.Conv1(x)
        return self.Conv2(x)

class KANv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        degree=3,
        drop_rate=0.1,
    ):
        super().__init__()
        self.dw_conv = nn.Conv3d(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            padding,
            groups=in_channels,
            bias=False,
        )
        self.norm = nn.InstanceNorm3d(in_channels)
        self.base_activation = nn.SiLU()
        self.base_pw = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        self.kan_pw = nn.Conv3d(
            in_channels * (degree + 1), out_channels, kernel_size=1, bias=False
        )
        self.register_buffer(
            "arange",
            torch.arange(degree + 1).view(1, 1, degree + 1, 1, 1, 1),
        )
        self.drop = nn.Dropout3d(drop_rate)
        self.out_activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.norm(self.dw_conv(x))
        base_out = self.base_pw(self.base_activation(x))
        x = torch.clamp(torch.tanh(x), -1 + 1e-6, 1 - 1e-6)
        basis = torch.cos(x.acos().unsqueeze(2) * self.arange)
        batch, channels, degree, depth, height, width = basis.shape
        basis = basis.reshape(batch, channels * degree, depth, height, width)
        out = base_out + self.kan_pw(basis)
        return self.out_activation(self.drop(out))


class kmireg(nn.Module):
    def __init__(self, size=(160, 192, 160), in_channel=1, first_channel=32):
        super().__init__()
        channels = first_channel
        self.encoder11 = KANv(in_channel, channels, stride=2)
        self.encoder12 = KANv(channels, channels, stride=2)
        self.encoder13 = KANv(channels, channels, stride=2)
        self.encoder21 = KANv(in_channel, channels, stride=2)
        self.encoder22 = KANv(channels, channels, stride=2)
        self.encoder23 = KANv(channels, channels, stride=2)
        self.media_enc = KANv(channels, channels, stride=2)
        self.media = MKANv(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=2,
            padding=1,
            degree=3,
        )       
        self.kmja = KMJA(channels, channels, channels)
        self.decoder3 = DecoderBlock(2 * channels, channels)
        self.decoder2 = DecoderBlock(2 * channels, channels)
        self.decoder1 = DecoderBlock(2 * channels, channels)
        self.gate_x = nn.Parameter(torch.tensor(0.1))
        self.gate_y = nn.Parameter(torch.tensor(0.1))
        self.transformer = nn.ModuleList(
            [SpatialTransformer([value // 2**level for value in size]) for level in range(3)]
        )
        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)

    def forward(self, x, y):
        fx1 = self.encoder11(x)
        fy1 = self.encoder21(y)
        sup_x = self.media(fx1, other_x=fy1)
        sup_y = self.media(fy1, other_x=fx1)

        fx2 = self.encoder12(fx1) + self.gate_x * sup_x
        fy2 = self.encoder22(fy1) + self.gate_y * sup_y
        fx3 = self.encoder13(fx2)
        fy3 = self.encoder23(fy2)
        sup_x = self.media_enc(sup_x)
        sup_y = self.media_enc(sup_y)

        coarse_feat = self.kmja(fx3, fy3, sup_x, sup_y)
        flow_all = self.decoder3(coarse_feat)
        flow_all = self.up(2 * flow_all)

        warped_x = self.transformer[2](fx2, flow_all)
        flow = self.decoder2(warped_x, fy2)
        flow_all = self.transformer[2](flow_all, flow) + flow
        flow_all = self.up(2 * flow_all)

        warped_x = self.transformer[1](fx1, flow_all)
        flow = self.decoder1(warped_x, fy1)
        flow_all = self.transformer[1](flow_all, flow) + flow
        flow_all = self.up(2 * flow_all)

        return self.transformer[0](x, flow_all), flow_all
