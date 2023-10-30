import numpy as np
import matplotlib.pyplot as plt

n = [0, 1, 2, 3, 4, 5, 6, 8, 10, 20]
# TiAlTa, E*, T = 100˚
n100 = [107.98086, 128.77345, 129.83005, 123.704216, 119.43117, 119.78103, 119.8425, 120.29419, 121.057365, 121.02052]
εn100 = [52.252518, 44.448933, 35.05853, 35.810158, 41.508427, 42.551655, 42.826797, 43.019463, 43.65513, 44.754738]
# TiAlTa, E*, T = 200˚
n200 = [108.73268, 128.22731, 129.9833, 124.54512, 118.53557, 119.37834, 119.91793, 119.599785, 121.14458, 121.09966]
εn200 = [50.40294, 45.410503, 35.427982, 35.49514, 41.152905, 42.339294, 42.827354, 43.35869, 43.716194, 44.876877]
# TiAlTa, T = 250˚
n250 = [67.35, 79.05415, 81.262184, 81.65729, 83.02792, 83.415115, 84.349846, 84.3668, 84.86247, 86.60203]
εn250 = [47.370438, 39.15883, 37.265903, 37.256607, 37.454945, 37.081303, 37.238724, 37.96447, 38.57067, 39.733036]

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
