import matplotlib.pyplot as plt
import json

def visualize_results(results, path='../../plots/'):
    label = json.dumps(results["metadata"], indent=2)
    for key, values in results.items():
        if key == "metadata": continue
        plt.figure()
        plt.plot(values, label=label, color="indigo")
        plt.title(key)
        plt.xlabel('Epoch')
        plt.grid(True)
        legend = plt.legend()
        for handle in legend.legendHandles:
            handle.set_visible(False)
        plt.savefig(path + key + '.png')
        plt.close()