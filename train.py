import argparse
from train_fun.train_VSR import train_vsr
from train_fun.train_SISR import train_sisr


parser = argparse.ArgumentParser(description="Train")
parser.add_argument("MODE", help="train mode", type=str)
parser.add_argument("cfg", help="cfg", type=str)
parser.add_argument("--model_path", default=None, type=str)

args = parser.parse_args()

if __name__ == "__main__":
    MODE = args.MODE
    cfg = args.cfg
    model_path = args.model_path

    if MODE == "VSR":
        if model_path == None:
            train_vsr(cfg)
        else:
            train_vsr(cfg, model_path)
    elif MODE == "SISR":
        train_sisr(cfg)