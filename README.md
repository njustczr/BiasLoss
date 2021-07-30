# BiasLoss
Bias Loss for Mobile Neural Networks 
```
from bias_loss import BiasLoss
import torch
# batch_size=3 num_classes=5
inputs = torch.randn(3,5,requires_grad=True)
targets = torch.randint(0,5,(3))
various = torch.randn(3,1)
b_loss = BiasLoss(0.3,0.3)
b_loss(inputs, targets, various)
```
