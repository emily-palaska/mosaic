import matplotlib.pyplot as plt
import json

def train_plot(results, path='../../plots/'):
    metadata = json.dumps(results["metadata"], indent=2)
    keys = ["loss", "labels", "var", "loss_sim"]

    for key in keys:
        train_values = results["train"][key]
        val_values = results["val"][key]

        plt.figure()
        plt.plot(train_values, label="train", color='darkcyan')
        plt.plot(val_values, label="val", color='indigo')
        plt.plot([], label=metadata)
        plt.title(key)
        plt.xlabel('Epoch')
        plt.grid(True)
        legend = plt.legend(fontsize=10)
        legend.legendHandles[2].set_visible(False)
        plt.savefig(path + key + '.png')
        plt.close()