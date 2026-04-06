import torch
import torch.nn.functional as F

from src.configs.global_config import DEVICE
from src.resnet import resnet18


def load_model(device=DEVICE):

    model = resnet18(pretrained=True)
    model.to(device).eval()

    for p in model.parameters():
        p.requires_grad_(False)

    return model


def get_gradcam_config(model):
    """
    Returns the model-specific Grad-CAM configuration, which for ResNet-18 is the
    last residual block in layer4. No reshape transform is needed because
    the activations are already standard CNN feature maps [B, C, H, W].
    """
    target_layer = model.layer4[-1]
    reshape_transform = None
    return target_layer, reshape_transform


@torch.no_grad()
def predict_resnet_embeddings(model, dataloader, device=None):
    """Returns penultimate embeddings [N, D] on CPU for the model with D = 512"""
    model.eval()
    outs = []

    if device is None:
        device = next(model.parameters()).device

    for xb, _ in dataloader:
        xb = xb.to(device, non_blocking=True)

        # forward up to avgpool
        x = model.conv1(xb)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)

        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)

        x = model.avgpool(x)                 # [B, 512, 1, 1]
        x = torch.flatten(x, 1)              # [B, 512]  (penultimate)
        outs.append(x.detach().cpu())

    return torch.cat(outs, dim=0)            # [N, 512]


@torch.no_grad()
def predict_logits_and_accuracy(model, dataloader, device=DEVICE):
    logits_chunks = []
    correct = 0
    total = 0

    for xb, yb in dataloader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        logits = model(xb)
        logits_chunks.append(logits.detach().cpu())

        pred = logits.argmax(dim=1)
        correct += (pred == yb).sum().item()
        total += yb.numel()

    logits_all = torch.cat(logits_chunks, dim=0)  # [N,C] on CPU
    acc = correct / total
    return logits_all, acc


def transform_logits_to_probs(logits):
    return torch.softmax(logits, dim=1)


def transform_logits_to_preds(logits):
    """logits: [N,C] -> preds: [N]"""
    return logits.argmax(dim=1)


# measures Uncertainty of model predictions als Verteilung der Wahrscheinlichkeiten über alle Klassen (high: insecure; low: secure) 
def entropy_from_logits(logits: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    # logits: [B, K]
    p = F.softmax(logits, dim=1)
    p = torch.clamp(p, min=eps, max=1.0)
    H = -(p * torch.log(p)).sum(dim=1)  # [B]
    return H
