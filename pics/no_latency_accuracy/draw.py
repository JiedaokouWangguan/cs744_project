import numpy as np
import matplotlib.pyplot as plt


worker1 = (0.9789, 0.9350)
worker2 = (0.9789, 0.9357)

ind = np.arange(len(worker1))  # the x locations for the groups
ind = ind
width = 0.2  # the width of the bars

fig, ax = plt.subplots()
rects2 = ax.bar(ind - 0.5 * width, worker1, width, color='navy', label='Worker1')
rects3 = ax.bar(ind + 0.5 * width, worker2, width, color='green', label='Worker2')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Accuracy')
ax.set_xlabel('Optimizer')
ax.set_title('Accuracy by Optimizer')
ax.set_xticks(ind)
ax.set_xticklabels(('EASGD', 'Downpour SGD'))
ax.legend()

plt.savefig("no_latency_accuracy.png")
plt.show()
