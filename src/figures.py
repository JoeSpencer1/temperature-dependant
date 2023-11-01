import matplotlib.pyplot as plt

'''
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
# 500˚C/250˚C
Ti500E = [46.683464, 38.522774, 24.701038, 22.080563, 22.1474, 21.659185, 16.620623]
εTi500E = [6.765701, 10.513843, 11.301726, 10.384357, 9.72246, 11.2682905, 11.029994]
Ti500σ = [84.67531, 105.72215, 45.78838, 69.46997, 60.99529, 55.9422, 49.83204]
εTi500σ = [46.46603, 74.18298, 39.670254, 57.913757, 35.607227, 45.5547, 38.628445]
# 25˚C/250˚C
Ti25250E = [41.319565, 34.81656, 33.59124, 31.09554, 33.010456, 32.823563, 40.46569]
εTi25250E = [1.6541697, 6.365257, 4.6176863, 2.6873071, 3.4111104, 2.5400345, 8.93229]
Ti25250σ = [168.42062, 67.375565, 123.53137, 152.6323, 187.2847, 97.121, 52.3555]
εTi25250σ = [240.32188, 32.01899, 141.78886, 136.91542, 330.77194, 63.82302, 58.671406]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.errorbar(n, Ti25E, yerr = εTi25E, color = 'blue', label = "25˚C")
ax1.errorbar(n, Ti250E, yerr = εTi250E, color = 'red', label = "250˚C")
ax1.set_yscale('log')
ax1.set_ylim([0.5, 200])
ax1.set_xlim([0, 21])
ax1.set_xticks([0, 5, 10, 15, 20])
ax1.set_yticks([1, 3, 10, 30, 100])
ax1.set_yticklabels([1, 3, 10, 30, 100])
ax1.legend()
ax1.set_ylabel("MAPE (%)")
ax1.set_xlabel("Experimental training data size")
ax1.annotate("A: $E_{r}$", xy=(0.15, 0.95), xycoords="axes fraction",
              fontsize=12, ha="center",
              bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightgray"))

ax2.errorbar(n1, Ti25σ, yerr = εTi25σ, color = 'blue', label = "25˚C")
ax2.errorbar(n1, Ti250σ, yerr = εTi250σ, color = 'red', label = "25˚C")
ax2.set_yscale('log')
ax2.set_ylim([10, 600])
ax2.set_xlim([-0.5, 21])
ax2.set_xticks([0, 5, 10, 15, 20])
ax2.set_yticks([10, 20, 50, 100, 200, 500])
ax2.set_yticklabels([10, 20, 50, 100, 200, 500])
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
'''
'''
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.errorbar(n, Ti25250E, yerr = εTi25250E, color = 'blue', label = "25˚C")
ax1.errorbar(n, Ti250E, yerr = εTi250E, color = 'red', label = "250˚C")
ax1.errorbar(n, Ti500E, yerr = εTi500E, color = 'black', label = "500˚C")
ax1.set_yscale('log')
ax1.set_ylim([0.5, 200])
ax1.set_xlim([0, 21])
ax1.set_xticks([0, 5, 10, 15, 20])
ax1.set_yticks([1, 3, 10, 30, 100])
ax1.set_yticklabels([1, 3, 10, 30, 100])
ax1.legend()
ax1.set_ylabel("MAPE (%)")
ax1.set_xlabel("Experimental training data size")
ax1.annotate("A: $E_{r}$ (250˚C)", xy=(0.15, 0.95), xycoords="axes fraction",
              fontsize=12, ha="center",
              bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightgray"))

ax2.errorbar(n1, Ti25250σ, yerr = εTi25250σ, color = 'blue', label = "25˚C")
ax2.errorbar(n1, Ti250σ, yerr = εTi250σ, color = 'red', label = "250˚C")
ax2.errorbar(n1, Ti500σ, yerr = εTi500σ, color = 'black', label = "500˚C")
ax2.set_yscale('log')
ax2.set_ylim([10, 600])
ax2.set_xlim([-0.5, 21])
ax2.set_xticks([0, 5, 10, 15, 20])
ax2.set_yticks([10, 20, 50, 100, 200, 500])
ax2.set_yticklabels([10, 20, 50, 100, 200, 500])
ax2.legend()
ax2.set_ylabel("MAPE (%)")
ax2.set_xlabel("Experimental training data size")
plt.subplots_adjust(bottom=0.180)
fig.tight_layout()
ax2.annotate("B: $\sigma_{y}$ (250˚C)", xy=(0.15, 0.95), xycoords="axes fraction",
              fontsize=12, ha="center",
              bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightgray"))
plt.savefig("/Users/Joe/Desktop/figure2.jpeg", dpi=800, bbox_inches="tight")
plt.show()
'''

n = [1, 2, 3, 4, 5, 10, 20]
n1 = n
# 25˚C
Ti750E = [154.16795, 25.680262, 10.1573925, 7.744658, 5.137566, 2.6173031, 1.9064773]
εTi750E = [158.71921, 47.042423, 6.292157, 5.007044, 6.3082805, 1.3194628, 1.7983129]
Ti750σ = [537.51544, 187.94482, 16.950771, 38.089996, 26.567835, 22.038841, 34.741573]
εTi750σ = [822.8985, 455.4716, 22.714993, 39.45992, 35.60631, 25.618887, 42.27451]
# 250˚C
Ti25250500E = [44.90654, 41.499107, 39.673485, 39.662952, 38.275635, 44.390118, 45.897182]
εTi25250500E = [4.357474, 8.017073, 4.3895893, 5.539613, 4.685218, 15.8127165, 15.998017]
Ti25250500σ = [326.08405, 180.70842, 192.22061, 246.42114, 166.95808, 277.3227, 243.02325]
εTi25250500σ = [507.5874, 199.93362, 193.73439, 199.61105, 132.54443, 372.73825, 322.32126]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.errorbar(n, Ti25250500E, yerr = εTi25250500E, color = 'blue', label = "25, 250, 500˚C")
ax1.errorbar(n, Ti750E, yerr = εTi750E, color = 'red', label = "750˚C")
ax1.set_yscale('log')
ax1.set_ylim([0.5, 300])
ax1.set_xlim([0, 21])
ax1.set_xticks([0, 5, 10, 15, 20])
ax1.set_yticks([1, 3, 10, 30, 100])
ax1.set_yticklabels([1, 3, 10, 30, 100])
ax1.legend()
ax1.set_ylabel("MAPE (%)")
ax1.set_xlabel("Experimental training data size")
ax1.annotate("A: $E_{r}$", xy=(0.15, 0.95), xycoords="axes fraction",
              fontsize=12, ha="center",
              bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightgray"))

ax2.errorbar(n1, Ti25250500σ, yerr = εTi25250500σ, color = 'blue', label = "25, 250, 500˚C")
ax2.errorbar(n1, Ti750σ, yerr = εTi750σ, color = 'red', label = "750˚C")
ax2.set_yscale('log')
ax2.set_ylim([10, 1300])
ax2.set_xlim([-0.5, 21])
ax2.set_xticks([0, 5, 10, 15, 20])
ax2.set_yticks([10, 20, 50, 100, 200, 500])
ax2.set_yticklabels([10, 20, 50, 100, 200, 500])
ax2.legend()
ax2.set_ylabel("MAPE (%)")
ax2.set_xlabel("Experimental training data size")
plt.subplots_adjust(bottom=0.180)
fig.tight_layout()
ax2.annotate("B: $\sigma_{y}$", xy=(0.15, 0.95), xycoords="axes fraction",
              fontsize=12, ha="center",
              bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightgray"))
plt.savefig("/Users/Joe/Desktop/figure3.jpeg", dpi=800, bbox_inches="tight")
plt.show()