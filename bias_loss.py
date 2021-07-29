import torch
import torch.nn.functional as F


class BiasLoss(torch.nn.Module):
    def __init__(self, alpha=0.0, beta=0.0, reduction='mean'):
        super(BiasLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction

    @staticmethod
    def _one_hot(targets: torch.Tensor, n_classes: int):
        with torch.no_grad():
            targets = torch.zeros(targets.size(0), n_classes).scatter_(1,targets.data.unsqueeze(1),1)
        return targets

    def forward(self, inputs, targets, various):
        targets = BiasLoss._one_hot(targets, inputs.size(-1))
        log_softmax = F.log_softmax(inputs, -1)
        z = various.exp()*self.alpha - self.beta
        loss = -(z*targets*log_softmax).sum(-1)

        if self.reduction == 'sum':
            loss = loss.sum()
        if self.reduction == 'mean':
            loss = loss.mean()

        return loss






