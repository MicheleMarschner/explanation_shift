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


@torch.no_grad()
def predict_resnet_embeddings(model, dataloader, device="cpu"):
    """
    Returns penultimate embeddings [N, D] on CPU for THIS custom ResNet.

    Embedding = output after avgpool+flatten, BEFORE fc.
    For ResNet-18 on CIFAR, D should be 512.
    """
    model.eval()
    outs = []

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


def entropy_from_logits(logits: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    # logits: [B, K]
    p = F.softmax(logits, dim=1)
    p = torch.clamp(p, min=eps, max=1.0)
    H = -(p * torch.log(p)).sum(dim=1)  # [B]
    return H
