import torch
import torch.nn as nn
from einops import rearrange
from timm.models.swin_transformer import PatchEmbed, SwinTransformerBlock, PatchMerging

# depuis le code officiel de swin unet
class PatchExpanding(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x)

        return x


class SwinBlockStack(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size):
        super().__init__()
        self.blocks = nn.Sequential(*[
            SwinTransformerBlock(
                dim=dim,
                input_resolution=(None, None),
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if i % 2 == 0 else window_size // 2,
                mlp_ratio=4.0,
                qkv_bias=True,
                drop_path=0.1
            ) for i in range(depth)
        ])

    def forward(self, x):
        return self.blocks(x)
    
#depth c'est le nombre de 

class SwinUNet(nn.Module):
    def __init__(self, img_size=256, in_chans=3, num_classes=4, embed_dim=96, depths=[2, 2, 2, 2], num_heads=[3, 6, 12, 24]):
        super().__init__()

        # Patch Partition + Linear Embedding 
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=4,
            in_chans=in_chans,
            embed_dim=embed_dim
        )

        self.encoder1 = SwinBlockStack(embed_dim, depths[0], num_heads[0], window_size=7)
        self.down1 = PatchMerging(embed_dim)

        self.encoder2 = SwinBlockStack(embed_dim * 2, depths[1], num_heads[1], window_size=7)
        self.down2 = PatchMerging(embed_dim * 2)

        self.encoder3 = SwinBlockStack(embed_dim * 4, depths[2], num_heads[2], window_size=7)
        self.down3 = PatchMerging(embed_dim * 4)

        # Bottleneck successive
        self.bottleneck1 = SwinBlockStack(embed_dim * 8, 1, num_heads[3], window_size=7)
        self.bottleneck2 = SwinBlockStack(embed_dim * 8, 1, num_heads[3], window_size=7)

        # PatchExpanding for Decoder
        self.up3 = PatchExpanding((img_size // 32, img_size // 32), embed_dim * 8)
        self.decoder3 = SwinBlockStack(embed_dim * 4, depths[2], num_heads[2], window_size=7)

        self.up2 = PatchExpanding((img_size // 16, img_size // 16), embed_dim * 4)
        self.decoder2 = SwinBlockStack(embed_dim * 2, depths[1], num_heads[1], window_size=7)

        self.up1 = PatchExpanding((img_size // 8, img_size // 8), embed_dim * 2)
        self.decoder1 = SwinBlockStack(embed_dim, depths[0], num_heads[0], window_size=7)

        self.final_up = PatchExpanding((img_size // 4, img_size // 4), embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Couches Linear pour la réduction de dimension après concaténation
        self.concat_linear3 = nn.Linear(embed_dim * 8, embed_dim * 4)
        self.concat_linear2 = nn.Linear(embed_dim * 4, embed_dim * 2)
        self.concat_linear1 = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, x):
        B = x.shape[0]

        # Patch embedding
        x, (H, W) = self.patch_embed(x)

        # Encoder
        x1 = self.encoder1(x)
        x = self.down1(x1)

        x2 = self.encoder2(x)
        x = self.down2(x2)

        x3 = self.encoder3(x)
        x = self.down3(x3)

        # Bottleneck successive 
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)

        # Decoder
        x = self.up3(x)
        x = torch.cat([x, x3], dim=-1)
        x = self.concat_linear3(x)
        x = self.decoder3(x)

        x = self.up2(x)
        x = torch.cat([x, x2], dim=-1)
        x = self.concat_linear2(x)
        x = self.decoder2(x)

        x = self.up1(x)
        x = torch.cat([x, x1], dim=-1)
        x = self.concat_linear1(x)
        x = self.decoder1(x)

        x = self.final_up(x)
        x = self.head(x)
        x = x.view(B, H, W, -1).permute(0, 3, 1, 2)  # B, num_classes, H, W

        return x

