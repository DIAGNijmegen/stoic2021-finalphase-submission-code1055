import os
from argparse import ArgumentParser

def parse_gpu():
    parser = ArgumentParser()
    parser.add_argument("gpu", type=int)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
