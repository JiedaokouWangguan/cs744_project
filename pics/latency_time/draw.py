import numpy as np
import matplotlib.pyplot as plt


ps = (250, 1314)
worker1 = (244, 1290)
worker2 = (228, 1305)

ind = np.arange(len(ps))  # the x locations for the groups
ind = ind
width = 0.2  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind - width, ps, width, color='IndianRed', label='Parameter Server')
rects2 = ax.bar(ind, worker1, width, color='navy', label='Worker1')
rects3 = ax.bar(ind + width, worker2, width, color='green', label='Worker2')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Finish time (s)')
ax.set_xlabel('Optimizer')
ax.set_title('Finish Time (30ms latency) by Optimizer')
ax.set_xticks(ind)
ax.set_xticklabels(('EASGD', 'Downpour SGD'))
ax.legend()

plt.savefig("latency_time.png")
plt.show()