from parser import parameter_parser
from utils import tab_printer, graph_reader, feature_reader, target_reader
from appnp import APPNPTrainer

def main():
    """
    Parsing command line parameters, reading data, fitting an APPNP/PPNP and scoring the model.
    """
    args = parameter_parser()
    tab_printer(args)
    graph = graph_reader(args.edge_path)
    features = feature_reader(args.features_path)
    target = target_reader(args.target_path)
    trainer = APPNPTrainer(args, graph, features, target)
    trainer.fit()

if __name__ == "__main__":
    main()