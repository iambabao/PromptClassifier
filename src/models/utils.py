# -*- coding: utf-8 -*-

"""
@Author             : Bao
@Date               : 2020/10/9
@Desc               :
@Last modified by   : Bao
@Last modified date : 2021/4/25
"""

import torch
import torch.nn.functional as F


def ce_loss(logits, labels, mask=None):
    """

    Args:
        logits: (batch, ..., num_labels)
        labels: (batch, ...)
        mask: (batch, ..., num_labels)

    Returns:

    """

    num_labels = logits.shape[-1]
    loss = F.cross_entropy(logits.view(-1, num_labels), labels.view(-1).type(torch.long), reduction='none')
    if mask is not None:
        if torch.sum(mask) != 0:
            loss = torch.sum(mask.view(-1) * loss) / torch.sum(mask)
        else:
            loss = 0.0
    else:
        loss = torch.mean(loss)
    return loss


def pu_loss_with_ce(logits, labels, prior, mask=None):
    ones = torch.ones(logits.shape[:-1], requires_grad=False).to(logits.device)
    zeros = torch.zeros(logits.shape[:-1], requires_grad=False).to(logits.device)
    mask = torch.ones_like(labels, requires_grad=False).to(logits.device) if mask is None else mask

    p_risk = ce_loss(logits, ones, mask=mask & (labels == 1))
    u_risk = ce_loss(logits, zeros, mask=mask & (labels == 0))
    n_risk = u_risk - prior * ce_loss(logits, zeros, mask=mask & (labels == 1))
    loss = prior * p_risk + n_risk if n_risk >= 0 else -n_risk

    return loss
