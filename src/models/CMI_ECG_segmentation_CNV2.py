import torch
import torch.nn as nn


class CBR_1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=9, stride=1, padding=4):
        super().__init__()
        self.seq_list = [
            nn.Conv1d(in_channels, out_channels, kernel, stride, padding, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        ]

        self.seq = nn.Sequential(*self.seq_list)

    def forward(self, x):
        return self.seq(x)


class Unet_1D(nn.Module):
    def __init__(self, class_n, layer_n):
        super().__init__()

        ### ------- encoder -----------
        self.enc1_1 = CBR_1D(1, layer_n)
        self.enc1_2 = CBR_1D(layer_n, layer_n)
        self.enc1_3 = CBR_1D(layer_n, layer_n)

        self.enc2_1 = CBR_1D(layer_n, layer_n * 2)
        self.enc2_2 = CBR_1D(layer_n * 2, layer_n * 2)

        self.enc3_1 = CBR_1D(layer_n * 2, layer_n * 4)
        self.enc3_2 = CBR_1D(layer_n * 4, layer_n * 4)

        self.enc4_1 = CBR_1D(layer_n * 4, layer_n * 8)
        self.enc4_2 = CBR_1D(layer_n * 8, layer_n * 8)

        ### ------- decoder -----------
        self.upsample_3 = nn.ConvTranspose1d(
            layer_n * 8, layer_n * 8, kernel_size=8, stride=2, padding=3
        )
        self.dec3_1 = CBR_1D(layer_n * 4 + layer_n * 8, layer_n * 4)
        self.dec3_2 = CBR_1D(layer_n * 4, layer_n * 4)

        self.upsample_2 = nn.ConvTranspose1d(
            layer_n * 4, layer_n * 4, kernel_size=8, stride=2, padding=3
        )
        self.dec2_1 = CBR_1D(layer_n * 2 + layer_n * 4, layer_n * 2)
        self.dec2_2 = CBR_1D(layer_n * 2, layer_n * 2)

        self.upsample_1 = nn.ConvTranspose1d(
            layer_n * 2, layer_n * 2, kernel_size=8, stride=2, padding=3
        )
        self.dec1_1 = CBR_1D(layer_n * 1 + layer_n * 2, layer_n * 1)
        self.dec1_2 = CBR_1D(layer_n * 1, layer_n * 1)
        self.dec1_3 = CBR_1D(layer_n * 1, class_n)
        self.dec1_4 = CBR_1D(class_n, class_n)

    def forward(self, x):
        enc1 = self.enc1_1(x)
        enc1 = self.enc1_2(enc1)
        enc1 = self.enc1_3(enc1)

        enc2 = nn.functional.max_pool1d(enc1, 2)
        enc2 = self.enc2_1(enc2)
        enc2 = self.enc2_2(enc2)

        enc3 = nn.functional.max_pool1d(enc2, 2)
        enc3 = self.enc3_1(enc3)
        enc3 = self.enc3_2(enc3)

        enc4 = nn.functional.max_pool1d(enc3, 2)
        enc4 = self.enc4_1(enc4)
        enc4 = self.enc4_2(enc4)

        dec3 = self.upsample_3(enc4)
        dec3 = self.dec3_1(torch.cat([enc3, dec3], dim=1))
        dec3 = self.dec3_2(dec3)

        dec2 = self.upsample_2(dec3)
        dec2 = self.dec2_1(torch.cat([enc2, dec2], dim=1))
        dec2 = self.dec2_2(dec2)

        dec1 = self.upsample_1(dec2)
        dec1 = self.dec1_1(torch.cat([enc1, dec1], dim=1))
        dec1 = self.dec1_2(dec1)
        dec1 = self.dec1_3(dec1)
        out = self.dec1_4(dec1)

        return out
