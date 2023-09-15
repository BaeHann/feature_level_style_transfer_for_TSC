import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1



class Discriminator_ATT(nn.Module):
    """Discriminator model for source domain."""
    def __init__(self, patch_size, att_hid_dim, depth, heads, mlp_dim, num_class):
        """Init discriminator."""
        self.patch_size =  patch_size
        self.hid_dim = att_hid_dim
        self.depth= depth
        self.heads = heads
        self.mlp_dim = mlp_dim
        super(Discriminator_ATT, self).__init__()
        self.transformer1= Seq_Transformer(patch_size=self.patch_size, dim=att_hid_dim, depth=self.depth, heads= self.heads , mlp_dim=self.mlp_dim)
        self.DC = nn.Linear(att_hid_dim, num_class)
    def forward(self, input):
        """Forward the discriminator."""
        input = input * 1.0
        input.register_hook(grl_hook(1.2))
        # src_shape = [batch_size, seq_len, input_dim]
        input = input.view(input.size(0),-1, self.patch_size )
        features = self.transformer1(input)
        domain_output = self.DC(features)
        return domain_output


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x


class Seq_Transformer(nn.Module):
    def __init__(self, *, patch_size, dim, depth, heads, mlp_dim, channels=1, dropout=0., emb_dropout=0.):
        super().__init__()
        # num_patches = (seq_len // patch_size)  # ** 2
        patch_dim = channels * patch_size  # ** 2
        # assert num_patches > MIN_NUM_PATCHES, f'your number of patches ({num_patches}) is way too small for attention to be effective. try decreasing your patch size'

        # self.patch_size = patch_size

        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        # self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)

        self.to_cls_token = nn.Identity()

    def forward(self, forward_seq):
        x = self.patch_to_embedding(forward_seq)
        # print(x.shape)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        # x += self.pos_embedding[:, :(n + 1)]
        # x = self.dropout(x)
        x = self.transformer(x)
        c_t = self.to_cls_token(x[:, 0])
        return c_t