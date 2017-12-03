"""
Script for applying the data processing tasks
"""


# -*- coding: utf-8 -*-
import os
import logging
#from dotenv import find_dotenv, load_dotenv
from .download import DatasetDownload
from .preprocessing import Preprocessing
from dotenv import find_dotenv, load_dotenv


def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """

    logger = logging.getLogger(__name__)
    logger.info('start downloading data into raw:')

    # Set environment variables.
    load_dotenv(find_dotenv())
    DatasetDownload.URL = str(os.environ.get("URL"))
    DatasetDownload.USERNAME = str(os.environ.get("LOGINNAME"))
    DatasetDownload.PASSWORD = str(os.environ.get("LOGINPASSWORD"))

    DatasetDownload.download_all()
    logger.info('downloading was successfull')

    # Not implemented yet
    Preprocessing.preprocess()

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
