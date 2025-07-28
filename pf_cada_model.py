# pf_cada_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


# ---------- GRL (Gradient Reversal Layer) ----------
class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


def grad_reverse(x, alpha=1.0):
    return GradReverse.apply(x, alpha)


# ---------- FPN backbone: return feature map + flattened vector ----------
class BaseNetwork(nn.Module):
    """
    forward_maps additionally returns the pyramid feature map (used by capsules and classifier).
    Output fmap: [B, 512, 8, 9]
    """
    def __init__(self):
        super(BaseNetwork, self).__init__()
        self.conv1 = nn.Conv2d(4, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, padding=1)
        self.conv5 = nn.Conv2d(512, 1024, kernel_size=4, padding=1)
        self.conv6 = nn.Conv2d(1024, 512, kernel_size=4, padding=1)
        self.conv7 = nn.Conv2d(512, 256, kernel_size=4, padding=1)
        self.conv8 = nn.Conv2d(256, 64, kernel_size=1, padding=0)
        self.pl = nn.MaxPool2d(kernel_size=2, stride=2)

        # FPN head: fuse c3 with upsampled c9
        self.feature_pyramid = nn.Conv2d(320, 512, kernel_size=1)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward_maps(self, x):
        c1 = F.relu(self.conv1(x))
        c2 = F.relu(self.conv2(c1))
        c3 = F.relu(self.conv3(c2))
        c4 = F.relu(self.conv4(c3))
        c5 = F.relu(self.conv5(c4))
        c6 = F.relu(self.conv6(c5))
        c7 = F.relu(self.conv7(c6))
        c8 = F.relu(self.conv8(c7))
        c9 = self.pl(c8)  # approx [B, 64, 4, 4]
        up_c9 = F.interpolate(c9, size=c3.size()[2:], mode='bilinear', align_corners=False)
        merged = torch.cat([c3, up_c9], dim=1)  # [B, 256+64=320, 8, 9]
        pyr = F.relu(self.feature_pyramid(merged))  # [B, 512, 8, 9]
        return pyr

    def forward(self, x):
        fmap = self.forward_maps(x)        # [B, 512, 8, 9]
        flat = fmap.flatten(1)             # [B, 512*8*9]
        return fmap, flat


# ---------- Capsule Network ----------
def squash(s, dim=-1, eps=1e-9):
    sq_norm = (s ** 2).sum(dim=dim, keepdim=True)
    scale = sq_norm / (1.0 + sq_norm)
    return scale * s / torch.sqrt(sq_norm + eps)


class PrimaryCapsules(nn.Module):
    """
    Convert feature map to Primary Capsules with a single conv.
    Let num_caps_maps=C1 and caps_dim=D1,
    then #capsules = H * W * C1.
    """
    def __init__(self, in_channels=512, caps_dim=8, num_caps_maps=32, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.caps_dim = caps_dim
        self.num_caps_maps = num_caps_maps
        self.conv = nn.Conv2d(in_channels, num_caps_maps * caps_dim, kernel_size, stride, padding)

    def forward(self, x):
        # x: [B, 512, H, W]
        out = self.conv(x)  # [B, num_caps_maps*caps_dim, H, W]
        B, C, H, W = out.size()
        out = out.view(B, self.num_caps_maps, self.caps_dim, H, W)  # [B, C1, D1, H, W]
        out = out.permute(0, 3, 4, 1, 2).contiguous()               # [B, H, W, C1, D1]
        out = out.view(B, H * W * self.num_caps_maps, self.caps_dim)  # [B, N_caps, D1]
        return squash(out, dim=-1)


class DigitCapsules(nn.Module):
    """
    Emotion/Digit Capsules with dynamic routing.
    num_classes: #emotion classes (DEAP=2, SEED=3)
    """
    def __init__(self, num_primary, in_dim, num_classes=2, out_dim=16, routing_iters=3):
        super().__init__()
        self.num_primary = num_primary
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.out_dim = out_dim
        self.routing_iters = routing_iters
        # W: [1, Np, num_classes, out_dim, in_dim]
        self.W = nn.Parameter(0.01 * torch.randn(1, num_primary, num_classes, out_dim, in_dim))

    def forward(self, u):
        # u: [B, Np, in_dim]
        B = u.size(0)
        u = u.unsqueeze(2).unsqueeze(3)              # [B, Np, 1, 1, D_in]
        u_hat = torch.matmul(self.W, u).squeeze(-1)  # [1,Np,num_cls,D_out,D_in] x [B,Np,1,1,D_in]
        u_hat = u_hat.expand(B, -1, -1, -1)          # [B, Np, num_classes, out_dim]

        b = torch.zeros(B, self.num_primary, self.num_classes, device=u_hat.device)
        for _ in range(self.routing_iters):
            c = F.softmax(b, dim=2)               # [B, Np, num_classes]
            c = c.unsqueeze(-1)                   # [B, Np, num_classes, 1]
            s = (c * u_hat).sum(dim=1)           # [B, num_classes, out_dim]
            v = squash(s, dim=-1)                # [B, num_classes, out_dim]
            b = b + (u_hat * v.unsqueeze(1)).sum(dim=-1)  # [B, Np, num_classes]
        return v  # [B, num_classes, out_dim]


# ---------- Domain Discriminator ----------
class DomainDiscriminator(nn.Module):
    def __init__(self, caps_out_dim=16, num_classes=2, hidden=128):
        super().__init__()
        in_dim = num_classes * caps_out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 2)  # 2 domains: source / target
        )

    def forward(self, caps_out, alpha=1.0):
        # caps_out: [B, num_classes, caps_out_dim]
        x = caps_out.flatten(1)            # [B, num_classes*caps_out_dim]
        x = grad_reverse(x, alpha=alpha)   # GRL
        return self.net(x)                 # [B, 2]


# ---------- Mamba-style Classifier (selection + state-space recursion) ----------
class MambaClassifier(nn.Module):
    """
    Selection + state-space recursion following Eq.(4)(5) style:
    Input X ∈ R^{T×F}; per-timestep selection/encoding, then linear state recursion; output logits.
    """
    def __init__(self, in_dim, hidden=256, num_classes=2):
        super().__init__()
        self.selector = nn.Linear(in_dim, hidden, bias=True)  # Ws, bs
        self.encoder  = nn.Linear(in_dim, hidden, bias=True)  # Wk, bk
        self.Wk_c     = nn.Linear(hidden, hidden, bias=False) # Wk_c
        self.U        = nn.Linear(hidden, hidden, bias=False) # U
        self.state_b  = nn.Parameter(torch.zeros(hidden))
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, x_seq):
        # x_seq: [B, T, F]
        B, T, F = x_seq.shape
        k = torch.zeros(B, self.encoder.out_features, device=x_seq.device)
        for t in range(T):
            xt = x_seq[:, t, :]                      # [B, F]
            gate = torch.sigmoid(self.selector(xt))  # selection gate
            enc  = torch.tanh(self.encoder(xt))      # encoding
            h    = gate * enc                        # H_t
            k    = torch.sigmoid(self.Wk_c(k) + self.U(h) + self.state_b)  # state recursion
        logits = self.classifier(k)                  # [B, num_classes]
        return logits


# ---------- Overall Model ----------
class MyModel(nn.Module):
    """
    FPN features -> Capsules (for domain alignment) + Mamba (for classification).
    forward returns (class_logits, domain_logits).
    """
    def __init__(self, num_classes=2, t_steps=6, img_hw=(8, 9), primary_caps_maps=32):
        super(MyModel, self).__init__()
        self.t_steps = t_steps
        self.img_hw = img_hw
        self.base_network = BaseNetwork()

        # Infer flattened feature dim dynamically (avoid hard-coded numbers)
        with torch.no_grad():
            dummy = torch.zeros(1, 4, img_hw[0], img_hw[1])
            fmap, flat = self.base_network(dummy)
            feat_dim = flat.shape[1]   # typically 512*8*9=36864

        # Capsule network
        self.primary_caps = PrimaryCapsules(in_channels=512, caps_dim=8, num_caps_maps=primary_caps_maps)
        self.num_primary_caps = img_hw[0] * img_hw[1] * primary_caps_maps
        self.digit_caps = DigitCapsules(num_primary=self.num_primary_caps, in_dim=8,
                                        num_classes=num_classes, out_dim=16, routing_iters=3)
        self.domain_disc = DomainDiscriminator(caps_out_dim=16, num_classes=num_classes, hidden=128)

        # Mamba classifier
        self.mamba = MambaClassifier(in_dim=feat_dim, hidden=256, num_classes=num_classes)

    def _single_step(self, x):
        # x: [B, 4, H, W]
        fmap, flat = self.base_network(x)       # fmap: [B,512,8,9], flat: [B,F]
        pri = self.primary_caps(fmap)           # [B, Np, 8]
        emo = self.digit_caps(pri)              # [B, num_classes, 16]
        return flat, emo

    def forward(self, x1, x2, x3, x4, x5, x6, alpha=1.0):
        flats, emos = [], []
        for xi in [x1, x2, x3, x4, x5, x6]:
            flat, emo = self._single_step(xi)
            flats.append(flat)
            emos.append(emo)

        # Classification (Mamba)
        x_seq = torch.stack(flats, dim=1)         # [B, T, F]
        class_logits = self.mamba(x_seq)          # [B, num_classes]

        # Domain discrimination (avg capsule outputs -> GRL -> discriminator)
        emo_avg = torch.stack(emos, dim=1).mean(dim=1)  # [B, num_classes, 16]
        domain_logits = self.domain_disc(emo_avg, alpha=alpha)  # [B, 2]

        return class_logits, domain_logits
