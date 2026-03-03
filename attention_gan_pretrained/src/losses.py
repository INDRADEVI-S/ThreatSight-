import torch.nn.functional as F
import torch

def hinge_d_loss(real_logits, fake_logits):
    return torch.mean(F.relu(1. - real_logits)) + torch.mean(F.relu(1. + fake_logits))

def hinge_g_loss(fake_logits):
    return -torch.mean(fake_logits)
