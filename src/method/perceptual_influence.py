import numpy as np
import matplotlib.pyplot as plt


experiments = {  # With transfer learning
    "VGG-19 ImageNet - block3_conv2": [  # 2030301631
        {"epoch": 1, "MSE": 0.0002040776889771223, "VGG_loss": 576.0071411132812},
        {"epoch": 5, "MSE": 9.793945355340838e-05, "VGG_loss": 433.1810302734375},
        {"epoch": 10, "MSE": 9.700160444481298e-05, "VGG_loss": 422.34844970703125},
    ],
    "VGG-19 ImageNet - block5_conv4": [  # 3001474124
        {"epoch": 1, "MSE": 0.0002935976954177022, "VGG_loss": 0.7757400274276733},
        {"epoch": 5, "MSE": 0.0001266493636649102, "VGG_loss": 0.5371675491333008},
        {"epoch": 10, "MSE": 0.00012420525308698416, "VGG_loss": 0.5127434134483337},
    ],
    "VGG-19 Medical - block3_conv2": [  # 255671025
        {"epoch": 1, "MSE": 0.002442130818963051, "VGG_loss": 737.3931884765625},
        {"epoch": 5, "MSE": 0.00017402890080120414, "VGG_loss": 192.37799072265625},
        {"epoch": 10, "MSE": 0.00016109268472064286, "VGG_loss": 163.57308959960938},
    ],
    "VGG-19 Medical - block5_conv4": [  # 22099157
        {"epoch": 1, "MSE": 0.00027333313482813537, "VGG_loss": 18.797935485839844},
        {"epoch": 5, "MSE": 0.00011429319420130923, "VGG_loss": 9.153958320617676},
        {"epoch": 10, "MSE": 0.00010968669812427834, "VGG_loss": 8.193005561828613},
    ],
}


lambdas = np.logspace(-7, 0, 500)

plt.figure(figsize=(10, 6))

for label, epoch_list in experiments.items():
    epoch_data = epoch_list[-1]
    MSE = epoch_data["MSE"]
    VGG_loss = epoch_data["VGG_loss"]

    percentual_influence = (lambdas * VGG_loss / (MSE + (lambdas * VGG_loss))) * 100
    plt.plot(lambdas, percentual_influence, label=label)

    scatter_lambdas = [10**-7, 10**-6, 10**-5, 10**-4, 10**-3, 10**-2, 10**-1, 1]
    scatter_values = [
        (l * VGG_loss / (MSE + (l * VGG_loss))) * 100 for l in scatter_lambdas
    ]
    plt.scatter(scatter_lambdas, scatter_values)

plt.xscale("log")
plt.xlabel(r"$\lambda$ (in logarithmic scale)", fontsize=12)
plt.ylabel("Percentage of perceptual loss influence (%)", fontsize=12)
plt.ylim(0, 110)
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend(title="Experiments", fontsize=9)
plt.tight_layout()

plt.savefig("lambda_study_last_epoch_comparison.eps", format="eps", dpi=300)
plt.show()
