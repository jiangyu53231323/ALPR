import numpy as np
from tensorboardX import SummaryWriter

writer = SummaryWriter('runs/another_scalar_example')
for i in range(10):
    writer.add_scalar('quadratic', i ** 3, global_step=i)
    writer.add_scalar('exponential', 3 ** i, global_step=i)
