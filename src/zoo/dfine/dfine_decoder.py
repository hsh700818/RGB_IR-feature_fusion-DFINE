import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from ...core import register
from .utils import MLP, inverse_sigmoid
from .box_ops import rbox_to_voffset, voffset_to_rbox 

__all__ = ['DFINETransformer']

class O2DFINEHead(nn.Module):
    def __init__(self, hidden_dim, num_bins):
        super().__init__()
        self.num_bins = num_bins
        self.reg_conf = nn.Linear(hidden_dim, 6 * num_bins)

    def forward(self, x):
        out = self.reg_conf(x)
        return out.view(out.shape[:-1] + (6, self.num_bins))

@register()
class DFINETransformer(nn.Module):
    __inject__ = [] 
    __share__ = ['num_classes']

    def __init__(self, 
                 num_layers=6, 
                 hidden_dim=256, 
                 num_bins=16, 
                 num_classes=5,
                 eval_idx=-1,
                 **kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_bins = num_bins
        self.num_classes = num_classes
        
        self.reg_head = nn.ModuleList([O2DFINEHead(hidden_dim, num_bins) for _ in range(num_layers)])
        self.cls_head = nn.ModuleList([nn.Linear(hidden_dim, num_classes) for _ in range(num_layers)])

        n = torch.arange(num_bins)
        mid = (num_bins - 1) / 2
        a, c = 2.0, 1.0
        sgn = torch.sign(n - mid)
        weights = torch.where(
            (n == 0) | (n == num_bins - 1), 
            torch.tensor(2 * a), 
            c * ((1 + a/c)**(2 * torch.abs(n - mid) / (num_bins - 2)) - 1)
        )
        self.register_buffer('adr_weights', sgn * weights)

    def forward(self, feats, targets=None):
        if isinstance(feats, (list, tuple)):
            memory = feats[-1]
        else:
            memory = feats
            
        bs = memory.shape[0]
        device = memory.device

        hs = torch.zeros(bs, 300, self.hidden_dim, device=device)
        curr_rbox = torch.zeros(bs, 300, 5, device=device) 
        curr_rbox[..., 2:4] = 0.1 

        output_classes = []
        output_distributions = []
        
        for i in range(self.num_layers):
            level_feat = F.adaptive_avg_pool2d(memory, (1, 1)).flatten(1)
            hs = hs + level_feat.unsqueeze(1)
            logits = self.reg_head[i](hs)
            probs = F.softmax(logits, dim=-1)
            offsets = torch.sum(probs * self.adr_weights, dim=-1)
            curr_rbox = voffset_to_rbox(curr_rbox, offsets)
            class_logits = self.cls_head[i](hs)
            output_classes.append(class_logits)
            output_distributions.append(logits)

        out = {
            'pred_logits': output_classes[-1], 
            'pred_boxes': curr_rbox, 
            'pred_reg_logits': output_distributions[-1],
            'aux_outputs': [
                {'pred_logits': a, 'pred_boxes': curr_rbox, 'pred_reg_logits': d} 
                for a, d in zip(output_classes[:-1], output_distributions[:-1])
            ]
        }
        return out