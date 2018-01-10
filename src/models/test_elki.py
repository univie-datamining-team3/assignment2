# -*- coding: utf-8 -*-
import os
import sys
from io import StringIO
import argparse
import logging
import pandas as pd
sys.path.append(os.path.join(os.getcwd(), os.pardir, 'src'))

from models.cluster import run_elki
from utils.utilities import str2bool


FLAGS = None
def main():
    data = pd.read_csv(FLAGS.data_dir, sep=";")
    results = run_elki(data)
    print(type(results))
    print(results)
    print("data.shape: ", data.shape)
    print("results.shape: ", results.shape)

    #results = pd.read_csv(StringIO(results), sep=" ", comment='#')
    #results = pd.read_csv(results)

    results.to_csv("elki_test.csv", index=False, sep=";")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str,
                        help="Directory for reading data")
    parser.add_argument('--save_results',
                        type=str2bool,
                        default="True",
                        help='Set true, if results should be saved')
    parser.add_argument('--iterations',
                        type=int,
                        default=100,
                        help='Set true, if results should be saved')
    FLAGS, unparsed = parser.parse_known_args()
    main()
