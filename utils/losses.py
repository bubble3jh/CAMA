import torch
import torch.nn as nn


class Entropy(nn.Module):
    def __init__(self):
        super(Entropy, self).__init__()

    def __call__(self, logits):
        return -(logits.softmax(1) * logits.log_softmax(1)).sum(1)


class I2TLoss(nn.Module):
    def __init__(self):
        super(I2TLoss, self).__init__()

    def __call__(self, logits, img_feats, text_norm_feats):
        labels = torch.argmax(logits.softmax(1), dim=1)
        loss = 0.0
        for l in torch.unique(labels, sorted = True).tolist():
            img_idx_embeddings = img_feats[labels == l]
            mean_feats = img_idx_embeddings.mean(0).type(text_norm_feats.dtype)
            dist = torch.matmul(mean_feats.unsqueeze(0), text_norm_feats[l].unsqueeze(0).t()).mean()
            loss += dist
        return loss / len(torch.unique(labels))
    
class InterMeanLoss(nn.Module):
    def __init__(self):
        super(InterMeanLoss, self).__init__()
        
    def __call__(self, logits, img_feats):
        labels = torch.argmax(logits.softmax(1), dim=1)
        mean_feats = []
        for l in torch.unique(labels, sorted = True).tolist():
            img_idx_embeddings = img_feats[labels == l]
            mean = img_idx_embeddings.mean(0)
            mean_feats.append(mean / mean.norm())

        cosine_sim_matrix = torch.matmul(torch.stack(mean_feats), torch.stack(mean_feats).t())
        loss = 1 - cosine_sim_matrix
        loss.fill_diagonal_(0)
        return loss.sum()


class SymmetricCrossEntropy(nn.Module):
    def __init__(self, alpha=0.5):
        super(SymmetricCrossEntropy, self).__init__()
        self.alpha = alpha

    def __call__(self, x, x_ema):
        return -(1-self.alpha) * (x_ema.softmax(1) * x.log_softmax(1)).sum(1) - self.alpha * (x.softmax(1) * x_ema.log_softmax(1)).sum(1)


class AugCrossEntropy(nn.Module):
    def __init__(self, alpha=0.5):
        super(AugCrossEntropy, self).__init__()
        self.alpha = alpha

    def __call__(self, x, x_aug, x_ema):
        return -(1-self.alpha) * (x.softmax(1) * x_ema.log_softmax(1)).sum(1) \
                  - self.alpha * (x_aug.softmax(1) * x_ema.log_softmax(1)).sum(1)


class SoftLikelihoodRatio(nn.Module):
    def __init__(self, clip=0.99, eps=1e-5):
        super(SoftLikelihoodRatio, self).__init__()
        self.eps = eps
        self.clip = clip

    def __call__(self, logits):
        probs = logits.softmax(1)
        probs = torch.clamp(probs, min=0.0, max=self.clip)
        return - (probs * torch.log((probs / (torch.ones_like(probs) - probs)) + self.eps)).sum(1)


class GeneralizedCrossEntropy(nn.Module):
    """ Paper: https://arxiv.org/abs/1805.07836 """
    def __init__(self, q=0.8):
        super(GeneralizedCrossEntropy, self).__init__()
        self.q = q

    def __call__(self, logits, targets=None):
        probs = logits.softmax(1)
        if targets is None:
            targets = probs.argmax(dim=1)
        probs_with_correct_idx = probs.index_select(-1, targets).diag()
        return (1.0 - probs_with_correct_idx ** self.q) / self.q


# ──────────────────────────────────────────────────────────────────────────────
# Projected Evidence-Gated losses (Hypothesis-driven improvements)
# ──────────────────────────────────────────────────────────────────────────────

def _text_projection_matrix(text_feats):
    """Return U (D×k) whose columns are the orthonormal basis of the text subspace.

    text_feats: (C, D) L2-normalised text prototypes.
    Returns U: (D, k) with k = rank of text_feats (≤ C).
    """
    # SVD on text_feats.T → (D, C).  U columns span the text subspace.
    U, _, _ = torch.linalg.svd(text_feats.float().T, full_matrices=False)
    return U  # (D, k)


class ProjectedInterMeanLoss(nn.Module):
    """InterMeanLoss restricted to the subspace spanned by text prototypes (Component 1).

    Projects img_feats onto the text subspace before computing inter-class
    prototype separation, preventing off-text dimension expansion.

    Args (call):
        logits      : (N, C) raw logits from the model.
        img_feats   : (N, D) pre-normalisation image features.
        text_feats  : (C, D) L2-normalised text prototypes.
    """

    def __init__(self):
        super().__init__()

    def __call__(self, logits, img_feats, text_feats):
        U = _text_projection_matrix(text_feats)          # (D, k)
        # v_parallel = img_feats @ U @ U.T  (N, D)
        v_par = img_feats.float() @ (U @ U.T)

        labels = torch.argmax(logits.softmax(1), dim=1)
        mean_feats = []
        for l in torch.unique(labels, sorted=True).tolist():
            emb = v_par[labels == l]
            mean = emb.mean(0)
            norm = mean.norm()
            mean_feats.append(mean / (norm + 1e-8))

        if len(mean_feats) < 2:
            return logits.new_zeros(1).squeeze()

        stacked = torch.stack(mean_feats)
        sim = torch.matmul(stacked, stacked.T)
        loss = 1.0 - sim
        loss.fill_diagonal_(0.0)
        return loss.sum()


class GatedI2TLoss(nn.Module):
    """I2TLoss with optional evidence gating — skips low-s_max samples (Component 2).

    Args (call):
        logits          : (N, C) raw logits.
        img_feats       : (N, D) pre-normalisation image features.
        text_norm_feats : (C, D) L2-normalised text prototypes.
        gate_mask       : (N,) bool tensor; True = admitted. None → admit all.
    """

    def __init__(self):
        super().__init__()

    def __call__(self, logits, img_feats, text_norm_feats, gate_mask=None):
        if gate_mask is not None:
            if gate_mask.sum() == 0:
                return logits.new_zeros(1).squeeze()
            logits    = logits[gate_mask]
            img_feats = img_feats[gate_mask]

        labels = torch.argmax(logits.softmax(1), dim=1)
        unique_labels = torch.unique(labels, sorted=True).tolist()
        if not unique_labels:
            return logits.new_zeros(1).squeeze()

        loss = 0.0
        for l in unique_labels:
            emb = img_feats[labels == l]
            mean = emb.mean(0).type(text_norm_feats.dtype)
            dist = torch.matmul(mean.unsqueeze(0), text_norm_feats[l].unsqueeze(0).T).mean()
            loss += dist
        return loss / len(unique_labels)


class GatedProjectedInterMeanLoss(nn.Module):
    """ProjectedInterMeanLoss with evidence gating (Components 1 + 2).

    Args (call):
        logits      : (N, C) raw logits.
        img_feats   : (N, D) pre-normalisation image features.
        text_feats  : (C, D) L2-normalised text prototypes.
        gate_mask   : (N,) bool tensor; True = admitted. None → admit all.
    """

    def __init__(self):
        super().__init__()

    def __call__(self, logits, img_feats, text_feats, gate_mask=None):
        if gate_mask is not None:
            if gate_mask.sum() == 0:
                return logits.new_zeros(1).squeeze()
            logits    = logits[gate_mask]
            img_feats = img_feats[gate_mask]

        U = _text_projection_matrix(text_feats)   # (D, k)
        v_par = img_feats.float() @ (U @ U.T)     # (N, D)

        labels = torch.argmax(logits.softmax(1), dim=1)
        mean_feats = []
        for l in torch.unique(labels, sorted=True).tolist():
            emb = v_par[labels == l]
            mean = emb.mean(0)
            norm = mean.norm()
            mean_feats.append(mean / (norm + 1e-8))

        if len(mean_feats) < 2:
            return logits.new_zeros(1).squeeze()

        stacked = torch.stack(mean_feats)
        sim = torch.matmul(stacked, stacked.T)
        loss = 1.0 - sim
        loss.fill_diagonal_(0.0)
        return loss.sum()
