import numpy as np
import torch
import torch.nn as nn

ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

ctc_loss = nn.CTCLoss(blank=len(ads) - 1, reduction='mean')
log_probs = torch.randn(18, 16, 35).log_softmax(2).detach().requires_grad_()
targets = torch.randint(1, 35, (16, 7), dtype=torch.long)
input_lengths = torch.full((16,), 18, dtype=torch.long)
target_lengths = torch.randint(4, 8, (16,), dtype=torch.long)
loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
loss.backward()

outputs = torch.randn(16, 19, 35)
out = [torch.topk(e, 1)[1] for e in outputs]
result = []
for b in range(16):
    output = out[b].squeeze()
    output = np.array([1,2,2,0,34,34,4,34,4,5,6,6,6])
    output = torch.from_numpy(output)
    res = []
    s = 0
    pre = -1
    for i in output:
        if i != pre:
            pre = i
            if i != 34:
                res.append(i)
    if len(res) > 7:
        res = res[:7]
    result.append(res)

for j in range(1):
    pass

print(out)
