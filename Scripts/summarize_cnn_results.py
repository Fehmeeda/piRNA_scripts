import json, glob
import numpy as np

files = glob.glob("results_cnn/*.json")

for species in ["Human","Mouse","Drosophila"]:
    accs = []
    aucs = []

    for f in files:
        if species in f:
            d = json.load(open(f))
            accs.append(d["test"]["acc"])
            aucs.append(d["test"]["auc"])

    print(f"{species}: ACC={np.mean(accs):.4f}, AUC={np.mean(aucs):.4f}")