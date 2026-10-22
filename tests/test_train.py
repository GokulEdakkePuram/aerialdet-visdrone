"""Partial weight transfer."""

from __future__ import annotations

import torch
from ultralytics.nn.tasks import DetectionModel

from aerialdet.train import reset_layers_after


def _layer(key: str) -> int:
    return int(key.split(".")[1])


def test_reset_layers_after_leaves_earlier_layers_alone():
    model = DetectionModel("yolo11n.yaml", verbose=False)
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(0.5)  # stand-in for pretrained weights

    n = reset_layers_after(model, 16)

    params = dict(model.named_parameters())
    kept = [k for k in params if _layer(k) <= 16]
    reset = [k for k in params if _layer(k) > 16 and params[k].dim() > 1]
    assert n > 0
    assert all(torch.all(params[k] == 0.5) for k in kept)
    assert reset and not any(torch.all(params[k] == 0.5) for k in reset)
