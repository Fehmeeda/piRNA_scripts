import json, glob
import numpy as np

files = glob.glob("results_cnn_7/*.json")

for species in ["Human","Mouse","Drosophila"]:
    accs = []
    aucs = []
    precison = []
    recall = []

    for f in files:
        if species in f:
            d = json.load(open(f))
            accs.append(d["test"]["acc"])
            aucs.append(d["test"]["auc"])
            precison.append(d["test"]["precision"])
            recall.append(d["test"]["recall"])


    print(f"{species}: ACC={np.mean(accs):.4f}, AUC={np.mean(aucs):.4f}, PRECISION={np.mean(precison):.4f}, RECALL={np.mean(recall):.4f}")