# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

easgd_quanti_x = [1, 2, 3, 4, 5]
easgd_x = [1, 2, 3, 4, 5]
easgd_quanti_y = [0.9789, 0.9788, 0.9789, 0.9787, 0.9790]
easgd_y = [0.978, 0.978, 0.978, 0.979, 0.978]
plt.figure(figsize=(8, 4))
plt.plot(easgd_quanti_x, easgd_quanti_y, color='SkyBlue', label='EASGD with Quantization')
plt.plot(easgd_x, easgd_y, color='IndianRed', label='EASGD')

plt.xlabel("Bandwidth (worker node)")
plt.xticks([1, 2, 3, 4, 5], ["5Mbs", "10Mbs", "20Mbs", "25Mbs", "50Mbs"])
plt.ylabel("Accuracy")
plt.title("Accuracy by Bandwidth")
plt.legend()
plt.savefig("downpur_easgd_quanti_accuracy.png")
plt.show()
