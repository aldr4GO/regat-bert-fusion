"""Axis 2: what alpha depends on (Sec. 7.2) + the paper-style free-weight baseline (Sec. 7.2b).

fused = alpha * s_sem_n + (1 - alpha) * s_ctx_n, one alpha in (0, 1) per query.
"""
import math

import torch
import torch.nn as nn


def fuse(sem_n, ctx_n, alpha):
    """alpha: scalar or [N] -> fused [N, K]."""
    alpha = torch.as_tensor(alpha, dtype=sem_n.dtype)
    if alpha.dim() == 1:
        alpha = alpha.unsqueeze(-1)
    return alpha * sem_n + (1 - alpha) * ctx_n


def logit(p):
    return math.log(p / (1 - p))


class AlphaMLP(nn.Module):
    def __init__(self, in_dim, hidden, dropout, alpha0, init_std):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
        last = self.net[-1]
        nn.init.normal_(last.weight, std=init_std)       # starts as the fixed-alpha baseline
        nn.init.constant_(last.bias, logit(alpha0))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class AlphaFixed(nn.Module):
    """Learned global scalar alpha = sigmoid(theta)."""

    def __init__(self, alpha0):
        super().__init__()
        self.theta = nn.Parameter(torch.tensor(logit(alpha0)))

    def forward(self, x):
        return self.theta.expand(x.shape[0])


class FusionModel(nn.Module):
    """dep in {fixed, Q, QS, S}. Inputs: q_emb [N, d], feats [N, 12] (standardised raw-score stats)."""

    def __init__(self, dep, d, n_feat, alpha0, fcfg):
        super().__init__()
        self.dep = dep
        mc = fcfg["mlp"]
        self.alpha_eps = fcfg["alpha_eps"]
        in_dim = {"Q": d, "S": n_feat, "QS": d + n_feat}.get(dep)
        if dep == "fixed":
            self.alpha_net = AlphaFixed(alpha0)
        else:
            self.alpha_net = AlphaMLP(in_dim, mc["hidden"], mc["dropout"], alpha0, mc["last_init_std"])
        self.log_T = nn.Parameter(torch.tensor(math.log(fcfg["train"]["init_T"])))
        self.bias = None   # only used with the BCE loss (ranking invariant)

    def inputs(self, q_emb, feats):
        if self.dep == "Q":
            return q_emb
        if self.dep == "S":
            return feats
        if self.dep == "QS":
            return torch.cat([q_emb, feats], -1)
        return q_emb   # fixed: only the batch size is used

    def alpha(self, q_emb, feats):
        z = self.alpha_net(self.inputs(q_emb, feats))
        return self.alpha_eps + (1 - 2 * self.alpha_eps) * torch.sigmoid(z)

    def forward(self, q_emb, feats, sem_n, ctx_n):
        a = self.alpha(q_emb, feats)
        return fuse(sem_n, ctx_n, a), a


class PaperFusion(nn.Module):
    """Row `paper`: fused = a * s_sem + b * s_ctx with free a, b (Eq. 5), trained with BCE (Eq. 6)."""

    def __init__(self, fcfg):
        super().__init__()
        pc = fcfg["paper"]
        self.a = nn.Parameter(torch.tensor(float(pc["init_a"])))
        self.b = nn.Parameter(torch.tensor(float(pc["init_b"])))
        self.bias = nn.Parameter(torch.tensor(0.0)) if pc["bias"] else None
        self.dep = "paper"

    def forward(self, q_emb, feats, sem_n, ctx_n):
        fused = self.a * sem_n + self.b * ctx_n
        # report the implied mixing weight a / (a + b) for analysis
        alpha = (self.a / (self.a + self.b)).expand(sem_n.shape[0])
        return fused, alpha
