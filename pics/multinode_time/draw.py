import numpy as np
import matplotlib.pyplot as plt


ps = (1691, 848, 399, 242)
worker1 = (851, 445, 225, 175)
worker2 = (837, 435, 222, 170)
worker3 = (1674, 846, 317, 167)
worker4 = (1683, 860, 394, 223)


ind = np.arange(len(ps))  # the x locations for the groups
ind = ind * 2
width = 0.2  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind - 2 * width, ps, width, color='SkyBlue', label='Parameter Server')
rects2 = ax.bar(ind - width, worker1, width, color='IndianRed', label='Worker1')
rects3 = ax.bar(ind, worker2, width, color='navy', label='Worker2')
rects4 = ax.bar(ind + width, worker3, width, color='brown', label='Worker3')
rects5 = ax.bar(ind + width * 2, worker4, width, color='coral', label='Worker4')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Finish time (s)')
ax.set_xlabel('Bandwidth (worker node)')
ax.set_title('EASGD Finish Time by Bandwidth')
ax.set_xticks(ind)
ax.set_xticklabels(('5Mbs', '10Mbs', '25Mbs', '50Mbs'))
ax.legend()


def autolabel(rects, xpos='center'):
    """
    Attach a text label above each bar in *rects*, displaying its height.

    *xpos* indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.
    """

    xpos = xpos.lower()  # normalize the case of the parameter
    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0.5, 'right': 0.57, 'left': 0.43}  # x_txt = x + w*off

    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()*offset[xpos], 1.01*height,
                '{}'.format(height), ha=ha[xpos], va='bottom')


# autolabel(rects1, "left")
# autolabel(rects2, "right")

plt.savefig("multi_node.png")
plt.show()