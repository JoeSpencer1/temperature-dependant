import numpy as np
import matplotlib.pyplot as plt

n = [1, 2, 3, 4, 5, 10, 20]
n1 = n
# 25˚C
Ti25E = [18.619205, 6.8593493, 6.0983825, 6.091887, 5.298505, 5.2244196, 3.6239312]
εTi25E = [1.6743975, 2.5646875, 1.9379541, 1.4813129, 1.9537387, 1.9212662, 2.01604]
Ti25σ = [34.068943, 18.617916, 43.272, 23.271488, 21.895184, 19.204937, 34.49958]
εTi25σ = [24.325975, 18.127224, 33.714092, 22.803173, 17.10482, 17.321714, 36.6192]
# 250˚C
Ti250E = [70.95709, 7.5985556, 8.114405, 1.6784289, 2.1151547, 1.8030748, 1.7651669]
εTi250E = [31.052818, 10.786752, 13.492649, 1.3040054, 1.0622171, 1.1414853, 1.182222]
Ti250σ = [261.21823, 25.825928, 49.60928, 17.916042, 17.266216, 16.254345, 47.430775]
εTi250σ = [270.2345, 23.495884, 65.91791, 15.266485, 21.114838, 16.953184, 59.20635]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.errorbar(n, Ti25E, yerr = εTi25E, color = 'blue', label = "25˚C")
ax1.errorbar(n, Ti25E, yerr = εTi25E, color = 'blue', label = "25˚C")
ax1.set_yscale('log')
ax1.set_ylim([1, 25])
ax1.set_xlim([0, 21])
ax1.set_xticks([0, 5, 10, 15, 20])
ax1.set_yticks([1, 5, 10, 15, 20])
ax1.set_yticklabels([1, 5, 10, 15, 25])
ax1.legend()
ax1.set_ylabel("MAPE (%)")
ax1.set_xlabel("Experimental training data size")
ax1.annotate("A: $E_{r}$", xy=(0.15, 0.95), xycoords="axes fraction",
              fontsize=12, ha="center",
              bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightgray"))

ax2.errorbar(n1, Ti25σ, yerr = εTi25σ, color = 'blue', label = "25˚C")
ax2.errorbar(n1, Ti25σ, yerr = εTi25σ, color = 'blue', label = "25˚C")
ax2.set_yscale('log')
ax2.set_ylim([1, 80])
ax2.set_xlim([-0.5, 21])
ax2.set_xticks([0, 5, 10, 15, 20])
ax2.set_yticks([1, 5, 10, 20, 40, 80])
ax2.set_yticklabels([1, 5, 10, 20, 40, 80])
ax2.legend()
ax2.set_ylabel("MAPE (%)")
ax2.set_xlabel("Experimental training data size")
plt.subplots_adjust(bottom=0.180)
fig.tight_layout()
ax2.annotate("B: $\sigma_{y}$", xy=(0.15, 0.95), xycoords="axes fraction",
              fontsize=12, ha="center",
              bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightgray"))
plt.savefig("/Users/Joe/Desktop/figure1.jpeg", dpi=800, bbox_inches="tight")
plt.show()