import numpy as np
from matplotlib import pyplot as plt
# import scienceplots
#
# plt.style.use('science')

alpha_conv = np.load("../output/new/alpha_conv.npy")

ue4_bins = np.array([6.03478131e-08])
um4_bins = np.linspace(0,0.5,11,True)
mass_bins = np.linspace(1,20,20,True)

fig, ax = plt.subplots()

for i in range(alpha_conv.shape[0]):
    for j in range(alpha_conv.shape[1]):
        for k in range(alpha_conv.shape[2]):
            ax.plot(range(alpha_conv.shape[3]), alpha_conv[i, j, k], label=f"Ue4_2: {ue4_bins[i]}, Um4_2: {um4_bins[j]}, Mass: {mass_bins[k]}")

            # var = alpha_conv[i, j, k] * (1 - alpha_conv[i, j, k]) / np.arange(1, alpha_conv.shape[3] + 1)
            # std = np.sqrt(var)
            # ax.fill_between(range(alpha_conv.shape[3]), alpha_conv[i, j, k] - std, alpha_conv[i, j, k] + std, alpha=0.2)

# ax.legend()
ax.set_xlabel("nFCE")
ax.set_ylabel("alpha")

plt.show()
