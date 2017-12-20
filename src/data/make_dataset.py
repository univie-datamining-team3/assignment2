"""
Script for applying the data processing tasks
"""


# -*- coding: utf-8 -*-
import os
import sys
import argparse
import logging
from dotenv import find_dotenv, load_dotenv
sys.path.append(os.path.join(os.getcwd(), os.pardir, 'src'))
from data.download import DatasetDownloader
from data.preprocessing import Preprocessor
from utils.utilities import str2bool
import numpy

FLAGS = None


def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../preprocessed).
    """

    logger = logging.getLogger(__name__)


    # Set environment variables.
    load_dotenv(find_dotenv())
    DatasetDownloader.URL = str(os.environ.get("URL"))
    DatasetDownloader.USERNAME = str(os.environ.get("LOGINNAME"))
    DatasetDownloader.PASSWORD = str(os.environ.get("LOGINPASSWORD"))


    if FLAGS.download:
        # Download data.
        logger.info('start downloading data into raw:')
        DatasetDownloader.download_all()
        logger.info('downloading was successfull')

    if FLAGS.preprocess:
        logger.info('start preprocessing data:')
        # Preprocess data. Store it in /data/preprocessed/preprocessed_data.dat.
        dfs = Preprocessor.preprocess([os.environ.get("KEY_RAPHAEL"),
                                       os.environ.get("KEY_MORITZ"),
                                       os.environ.get("KEY_LUKAS")],
                                      filename="preprocessed_data.dat")

        # Load dataframes from disk.
        # dfs = Preprocessor.restore_preprocessed_data_from_disk(filename="preprocessed_data.dat")

        logger.info('preprocessing was successful')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    parser = argparse.ArgumentParser()
    parser.add_argument('--download',
                        type=str2bool,
                        default="True",
                        help='Set true, if you want to download all data')
    parser.add_argument('--preprocess',
                        type=str2bool,
                        default="True",
                        help='Set true, if you want to apply the preprocessing')
    FLAGS, unparsed = parser.parse_known_args()
    main()
