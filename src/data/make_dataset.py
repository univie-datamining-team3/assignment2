"""
Script for applying the data processing tasks
"""


# -*- coding: utf-8 -*-
import os
import sys
import argparse
import logging
sys.path.append(os.path.join(os.getcwd(), os.pardir, 'src'))

from dotenv import find_dotenv, load_dotenv
from data.download import DatasetDownloader
from data.preprocessing import Preprocessor
from utils.utilities import str2bool

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
        tokens = [os.environ.get(alias) for alias in ["KEY_RAPHAEL", "KEY_MORITZ", "KEY_LUKAS"]]
        dfs = Preprocessor.preprocess(tokens,
                                      filename=FLAGS.file_name,
                                      distance_metric=FLAGS.distance_metric,
                                      use_individual_columns=FLAGS.use_individual_columns)

        # Load dataframes from disk.
        # dfs = Preprocessor.restore_preprocessed_data_from_disk(filename="preprocessed_data.dat")
#
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
    parser.add_argument('--use_individual_columns',
                        type=str2bool,
                        default="False",
                        help='Set true, if you want to apply the preprocessing and distance calculation to all individual columns')
    parser.add_argument('--distance_metric',
                        type=str,
                        default="euclidean",
                        help='specify which distance metric should be used, e.g. cityblock, sqeuclidean, dtw ... For a full list of all available choices see our documentation')
    parser.add_argument('--file_name',
                        type=str,
                        default="preprocessed_data.dat",
                        help='specify the name of the pickled output file. NOTE that all other file names will start with the specified name.')

    FLAGS, unparsed = parser.parse_known_args()
    main()
