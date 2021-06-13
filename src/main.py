""" Running the APPNP model."""

import numpy as np
import torch
from appnp import APPNPTrainer
from param_parser import parameter_parser
from utils import tab_printer, graph_reader, feature_reader, target_reader, attribute_reader

def main(alpha=0.0):
    """
    Parsing command line parameters, reading data, fitting an APPNP/PPNP and scoring the model.
    """
    args = parameter_parser()
    args.alpha = alpha
    torch.manual_seed(args.seed)
#   tab_printer(args)
    graph = graph_reader(args.edge_path)
    attributes = attribute_reader(args.attributes_path)
    features = np.array(attributes.drop([args.target], axis=1))
    target = np.array(attributes[args.target])

    trainer = APPNPTrainer(args, graph, features, target)
    acc = trainer.fit()
    return acc

if __name__ == "__main__":

    acc_list = []
    for alpha_c in [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99, 0.999]:
        acc_at_alpha = []
        for _ in range(10):
            acc_at_alpha.append(main(1.0-alpha_c))
        acc_list.append(sum(acc_at_alpha) / len(acc_at_alpha))
    print("overall accuracy:    ", max(acc_list), "    ", acc_list)
