import torch
import torch.nn as nn

class IoULoss(nn.Module):
    def __init__(self, eps=1e-6, reduction="mean"):
        super().__init__()
        self.eps=eps
        self.reduction=reduction

        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(f"Incorrect reduction argument {reduction}. Choose from 'none', 'mean', 'sum'")
        
    def forward(self, pred_boxes, target_boxes):
        pred_cx, pred_cy, pred_w, pred_h = pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 2], pred_boxes[:, 3]
        pred_x1 = pred_cx - pred_w/2
        pred_y1 = pred_cy - pred_h/2
        pred_x2 = pred_cx + pred_w/2
        pred_y2 = pred_cy + pred_h/2

        target_cx, target_cy, target_w, target_h = target_boxes[:, 0], target_boxes[:, 1], target_boxes[:, 2], target_boxes[:, 3]
        target_x1 = target_cx - target_w/2
        target_y1 = target_cy - target_h/2
        target_x2 = target_cx + target_w/2
        target_y2 = target_cy + target_h/2

        inter_x1=torch.max(pred_x1, target_x1)
        inter_y1=torch.max(pred_y1, target_y1)
        inter_x2=torch.min(pred_x2, target_x2)
        inter_y2=torch.min(pred_y2, target_y2)

        inter_w=torch.clamp(inter_x2-inter_x1, min=0)
        inter_h=torch.clamp(inter_y2-inter_y1, min=0)

        intersection=inter_w*inter_h

        pred_area=pred_w*pred_h
        target_area=target_w*target_h

        union=pred_area+target_area-intersection+self.eps

        iou=intersection/union
        loss=1-iou

        if self.reduction=='mean':
            return loss.mean()
        elif self.reduction=='sum':
            return loss.sum()
        else:
            return loss

                             
    
                             