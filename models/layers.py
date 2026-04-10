import torch
import torch.nn as nn

class CustomDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        if p<0.0 or p>1.0:
            raise ValueError(f"Value of p = {p}. Choose p from [0.0, 1.0]")
        self.p=p

    def forward(self, x):
        if self.training:
            if self.p==1.0:
                return torch.zeros_like(x)
            mask=torch.bernoulli(torch.full_like(x, 1-self.p))
            output=(x*mask)/(1-self.p)
            return output
        else:
            return x