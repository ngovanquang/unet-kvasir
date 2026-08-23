"""
U-Net tu cai cho phan doan anh y sinh (De tai 5 - Kvasir-SEG).

Kien truc encoder-decoder co skip connection, tham so hoa san hai truc ablation
ma de bai yeu cau, de khong phai sua kien truc giua tuan:

    up_mode   : "transpose" | "bilinear" | "unpool"
    skip_mode : "full"      | "half"     | "none"

QUAN TRONG: forward() tra ve LOGITS (khong co sigmoid).
    - BCE / Weighted BCE : dung nn.BCEWithLogitsLoss (on dinh so hoc hon)
    - Dice / Focal / Tversky : tu ap torch.sigmoid(logits) ben trong ham loss
    - Tinh metric : (torch.sigmoid(logits) > 0.5) voi nguong CO DINH 0.5

Khong dung thu vien segmentation dung san (segmentation_models_pytorch, ...)
dung nhu yeu cau cua de bai.
"""

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

UP_MODES = ("transpose", "bilinear", "unpool")
SKIP_MODES = ("full", "half", "none")


class DoubleConv(nn.Module):
    """(Conv 3x3 -> BN -> ReLU) x 2. Khoi co ban cua U-Net.

    Dung padding=1 de giu nguyen kich thuoc, khac ban goc 2015 (khong padding,
    phai crop skip). Neu muon bam sat bai bao thi bo padding va crop trong
    _merge_skip, nhung padding=1 de doi chieu kich thuoc hon nhieu.
    """

    def __init__(self, in_ch: int, out_ch: int, mid_ch: Optional[int] = None):
        super().__init__()
        mid_ch = mid_ch or out_ch
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    """Mot tang decoder: tang mau -> (ghep skip) -> DoubleConv.

    Ba kieu tang mau:
      transpose : ConvTranspose2d 2x2 stride 2 (co tham so hoc, de sinh
                  checkerboard artifact vi kernel 2 chia het cho stride 2
                  nhung van chong lan khi ket hop qua nhieu tang)
      bilinear  : Upsample bilinear + Conv 3x3 (khong co tham so o buoc tang
                  mau, thuong sach artifact hon)
      unpool    : MaxUnpool2d dung lai chi so tu MaxPool2d ben encoder.
                  MaxUnpool doi tensor dau vao CUNG SO KENH voi tensor da
                  duoc pool, nen phai co Conv 1x1 giam kenh truoc, roi Conv
                  3x3 sau de lam day cac o rong (unpool tra ve tensor thua).
    """

    def __init__(self, in_ch: int, out_ch: int, skip_ch: int, up_mode: str):
        super().__init__()
        if up_mode not in UP_MODES:
            raise ValueError(f"up_mode phai thuoc {UP_MODES}, nhan duoc {up_mode!r}")
        self.up_mode = up_mode
        self.skip_ch = skip_ch

        if up_mode == "transpose":
            self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        elif up_mode == "bilinear":
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
        else:  # unpool
            self.reduce = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
            self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
            self.fill = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.conv = DoubleConv(out_ch + skip_ch, out_ch)

    def forward(
        self,
        x: torch.Tensor,
        skip: Optional[torch.Tensor],
        indices: torch.Tensor,
        target_size: torch.Size,
    ) -> torch.Tensor:
        if self.up_mode == "unpool":
            x = self.reduce(x)
            x = self.unpool(x, indices, output_size=target_size)
            x = self.fill(x)
        else:
            x = self.up(x)

        # Phong khi kich thuoc lech 1 pixel do anh dau vao khong chia het.
        if x.shape[-2:] != target_size:
            x = F.interpolate(x, size=target_size, mode="nearest")

        if skip is not None:
            x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """U-Net tham so hoa.

    Args:
        in_channels   : 3 cho anh RGB Kvasir-SEG.
        num_classes   : 1 cho phan doan nhi phan (dung voi BCEWithLogitsLoss).
        base_channels : so kenh tang dau. 64 = ban goc (~31M tham so).
                        Dat 32 neu Colab het VRAM (~7.8M tham so).
        depth         : so tang downsample. 4 = ban goc.
        up_mode       : "transpose" | "bilinear" | "unpool".
        skip_mode     : "full" giu du skip;
                        "half" chi giu skip o cac tang SAU (do phan giai thap),
                               bo cac tang NONG (do phan giai cao) - dung de
                               chung minh vai tro cua thong tin do phan giai cao;
                        "none" bo hoan toan skip (thanh encoder-decoder thuan).
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        base_channels: int = 64,
        depth: int = 4,
        up_mode: str = "transpose",
        skip_mode: str = "full",
    ):
        super().__init__()
        if up_mode not in UP_MODES:
            raise ValueError(f"up_mode phai thuoc {UP_MODES}, nhan duoc {up_mode!r}")
        if skip_mode not in SKIP_MODES:
            raise ValueError(f"skip_mode phai thuoc {SKIP_MODES}, nhan duoc {skip_mode!r}")
        if depth < 1:
            raise ValueError("depth phai >= 1")

        self.depth = depth
        self.up_mode = up_mode
        self.skip_mode = skip_mode
        self.divisor = 2 ** depth

        enc_channels: List[int] = [base_channels * (2 ** i) for i in range(depth)]
        bottleneck_ch = base_channels * (2 ** depth)
        self.skip_flags = self._build_skip_flags(skip_mode, depth)

        # Encoder
        self.encoders = nn.ModuleList()
        prev = in_channels
        for ch in enc_channels:
            self.encoders.append(DoubleConv(prev, ch))
            prev = ch
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.bottleneck = DoubleConv(enc_channels[-1], bottleneck_ch)

        # Decoder: duyet tu tang sau nhat len tang nong nhat
        self.ups = nn.ModuleList()
        in_ch = bottleneck_ch
        for level in reversed(range(depth)):
            out_ch = enc_channels[level]
            skip_ch = enc_channels[level] if self.skip_flags[level] else 0
            self.ups.append(UpBlock(in_ch, out_ch, skip_ch, up_mode))
            in_ch = out_ch

        self.head = nn.Conv2d(enc_channels[0], num_classes, kernel_size=1)
        self._init_weights()

    @staticmethod
    def _build_skip_flags(skip_mode: str, depth: int) -> List[bool]:
        """Index 0 = tang nong nhat (do phan giai cao nhat)."""
        if skip_mode == "full":
            return [True] * depth
        if skip_mode == "none":
            return [False] * depth
        keep_from = depth - depth // 2  # depth=4 -> [F, F, T, T]
        return [level >= keep_from for level in range(depth)]

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _pad_to_divisor(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        h, w = x.shape[-2:]
        pad_h = (self.divisor - h % self.divisor) % self.divisor
        pad_w = (self.divisor - w % self.divisor) % self.divisor
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return x, (h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, (orig_h, orig_w) = self._pad_to_divisor(x)

        feats: List[torch.Tensor] = []
        indices: List[torch.Tensor] = []
        sizes: List[torch.Size] = []
        for encoder in self.encoders:
            x = encoder(x)
            feats.append(x)
            sizes.append(x.shape[-2:])
            x, idx = self.pool(x)
            indices.append(idx)

        x = self.bottleneck(x)

        for i, up in enumerate(self.ups):
            level = self.depth - 1 - i
            skip = feats[level] if self.skip_flags[level] else None
            x = up(x, skip, indices[level], sizes[level])

        logits = self.head(x)
        return logits[..., :orig_h, :orig_w]

    def config(self) -> dict:
        """Tra ve dict de ghi thang vao logs/runs.csv."""
        return {
            "up_mode": self.up_mode,
            "skip_mode": self.skip_mode,
            "depth": self.depth,
            "n_params": count_parameters(self),
        }


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    torch.manual_seed(42)
    x = torch.randn(2, 3, 256, 256)

    print(f"{'up_mode':<11}{'skip_mode':<11}{'output':<22}{'params':>12}")
    print("-" * 56)
    for up_mode in UP_MODES:
        for skip_mode in SKIP_MODES:
            model = UNet(up_mode=up_mode, skip_mode=skip_mode)
            with torch.no_grad():
                y = model(x)
            assert y.shape == (2, 1, 256, 256), y.shape
            print(f"{up_mode:<11}{skip_mode:<11}{str(tuple(y.shape)):<22}"
                  f"{count_parameters(model):>12,}")

    # Kich thuoc le: kiem tra co che pad/crop
    model = UNet(up_mode="unpool", skip_mode="half")
    y = model(torch.randn(1, 3, 250, 187))
    assert y.shape == (1, 1, 250, 187), y.shape
    print(f"\nAnh 250x187 -> {tuple(y.shape)}  (pad/crop hoat dong)")

    # Backward mot buoc de chac chan gradient chay qua ca ba kieu tang mau
    for up_mode in UP_MODES:
        model = UNet(up_mode=up_mode, base_channels=16)
        logits = model(torch.randn(1, 3, 64, 64))
        loss = nn.BCEWithLogitsLoss()(logits, torch.zeros_like(logits))
        loss.backward()
        n_none = sum(1 for p in model.parameters() if p.grad is None)
        print(f"backward {up_mode:<10} ok, so tham so khong co gradient: {n_none}")
