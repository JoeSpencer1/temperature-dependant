import numpy as np
import matplotlib.pyplot as plt

n = [0, 1, 2, 3, 4, 5, 10, 20]
# TiAlTa, E*, T = 100˚
n100 = []
εn100 = []
# TiAlTa, E*, T = 200˚
n200 = []
εn200 = []
# TiAlTa, T = 250˚
n250 = []
εn250 = []

fig, ax = plt.subplots()
ax.errorbar(n, n100, yerr = εn100, label = "TiAlTa: 100˚C")
ax.errorbar(n, n200, yerr = εn200, label = "TiAlTa: 200˚C")
ax.errorbar(n, n250, yerr = εn250, label = "TiAlTa: 250˚C (cross2)")
ax.set_yscale('linear')
ax.set_ylim([0, 250])
ax.set_xlim([-1, 21])
ax.set_xticks([0, 5, 10, 15, 20])
ax.set_yticks([0, 100, 200, 250])
ax.set_yticklabels([0, 100, 200, 250])
ax.legend()
ax.set_ylabel("$E^{\star}$, GPa")
ax.set_xlabel("Randomly selected training data set size ($n_{exp}$)")
plt.subplots_adjust(bottom=0.18)
plt.suptitle("$E^{\star}$: trained by 25˚C and 250˚C data", y=0.05, fontsize=16)
plt.savefig("/Users/Joe/Desktop/figure32.png")
plt.show()
