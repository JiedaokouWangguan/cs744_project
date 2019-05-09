# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

downpour_x = [5, 6, 7, 8, 9]
easgd_x = [1, 2, 3, 4, 5]
downpour_y = [0.9362, 0.9355, 0.9330, 0.9344, 0.9434]
easgd_y = [0.9778, 0.9789, 0.9789, 0.9790, 0.9789]
plt.figure(figsize=(8, 4))
plt.plot(downpour_x, downpour_y, color='SkyBlue', label='Downpour SGD')
plt.plot(easgd_x, easgd_y, color='IndianRed', label='EASGD')

plt.xlabel("Bandwidth (worker node)")
plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9], ["5Mbs", "10Mbs", "20Mbs", "25Mbs", "50Mbs", "80Mbs", "160Mbs", "320Mbs", "640Mbs"])
plt.ylabel("Finish Time (s)")
plt.title("Finish Time by Bandwidth")
plt.legend()
plt.savefig("downpur_easgd_accuracy.png")
plt.show()
