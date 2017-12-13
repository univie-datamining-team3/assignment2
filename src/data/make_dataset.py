"""
Script for applying the data processing tasks
"""


# -*- coding: utf-8 -*-
import os
import logging
from dotenv import find_dotenv, load_dotenv
from data.download import DatasetDownloader
from data.preprocessing import Preprocessor
import numpy


def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """

    logger = logging.getLogger(__name__)
    logger.info('start downloading data into raw:')

    # Set environment variables.
    load_dotenv(find_dotenv())
    DatasetDownloader.URL = str(os.environ.get("URL"))
    DatasetDownloader.USERNAME = str(os.environ.get("LOGINNAME"))
    DatasetDownloader.PASSWORD = str(os.environ.get("LOGINPASSWORD"))

    dfs = Preprocessor.preprocess([os.environ.get("KEY_RAPHAEL")])

    cd = dfs[os.environ.get("KEY_RAPHAEL")]["trips"][14]
    print(cd["sensor"].isnull().sum().sum())
    all_sensors_resampled = Preprocessor.downsample_time_series_per_category(cd["sensor"],
                                                                             categorical_colnames=["sensor"])
    print(all_sensors_resampled.isnull().sum().sum())

    # Download data.
    # DatasetDownloader.download_all()
    #logger.info('downloading was successfull')

    # Not implemented yet
    #dfs = Preprocessor.preprocess([os.environ.get("KEY_RAPHAEL"),
    #                               os.environ.get("KEY_MORITZ"),
    #                               os.environ.get("KEY_LUKAS")])


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
