# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

easgd_quanti_x = [1, 2, 3, 4, 5]
easgd_x = [1, 2, 3, 4, 5]
easgd_quanti_y = [477, 299, 236, 203, 187]
easgd_y = [923, 480, 307, 287, 219]
plt.figure(figsize=(8, 4))
plt.plot(easgd_quanti_x, easgd_quanti_y, color='SkyBlue', label='EASGD with Quantization')
plt.plot(easgd_x, easgd_y, color='IndianRed', label='EASGD')

plt.xlabel("Bandwidth (worker node)")
plt.xticks([1, 2, 3, 4, 5], ["5Mbs", "10Mbs", "20Mbs", "25Mbs", "50Mbs"])
plt.ylabel("Finish Time (s)")
plt.title("Finish Time by Bandwidth")
plt.legend()
plt.savefig("downpur_easgd_quanti_time.png")
plt.show()
