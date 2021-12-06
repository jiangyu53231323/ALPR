import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-4 * np.pi, 4 * np.pi)
y = np.sin(x)
plt.plot(x, y, '.-')

# plt保存要放在show前面，否则保存的就是一张空白图片
plt.show()
plt.savefig('pr_graph.png')
