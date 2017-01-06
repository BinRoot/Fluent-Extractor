import numpy as np
import matplotlib.pyplot as plt

with open('my_grid_predictions', 'r') as reader:
    str_vals = reader.readlines()
    vals = np.asarray(map(float, str_vals))
    vals = (vals - np.min(vals)) / np.ptp(vals)
    idx = 0
    plt.figure()
#    colors = iter(cm.rainbow(np.linspace(0, 1, len(100*100))))
    for w in np.arange(0, 1, 0.02):
        for h in np.arange(0, 1, 0.02):
            val = vals[idx]
            plt.scatter(w, h, color=str(val))
            idx += 1
    plt.show()
