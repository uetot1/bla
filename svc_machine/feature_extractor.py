import copy

import torch
from torch import nn

from models.experimental import attempt_load


FRONTEND_LAST_LAYER = 4


def make_yolo_teacher_and_clone(weights, device):
    teacher = attempt_load(weights, device=device, fuse=False).to(device).eval()
    if len(teacher.model) <= FRONTEND_LAST_LAYER:
        raise ValueError('The selected YOLO model does not expose layers 0..4')
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    clone = copy.deepcopy(teacher.model[:FRONTEND_LAST_LAYER + 1]).to(device)
    for parameter in clone.parameters():
        parameter.requires_grad_(True)
    return teacher, clone


@torch.no_grad()
def extract_teacher_feature(teacher, image):
    return teacher(image, cut_model=1, cutting_layer=FRONTEND_LAST_LAYER).detach()


def install_cloned_frontend(detector, state_dict):
    if state_dict is None:
        return False
    if not getattr(detector, 'pt', False):
        raise ValueError('A trained cloned front-end requires a PyTorch YOLO checkpoint')
    frontend = nn.Sequential(*list(detector.model.model.children())[:FRONTEND_LAST_LAYER + 1])
    frontend.load_state_dict(state_dict, strict=True)
    return True
