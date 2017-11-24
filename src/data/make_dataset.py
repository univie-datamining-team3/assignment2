"""
Script for applying the data processing tasks
"""



# -*- coding: utf-8 -*-
import os
import logging
#from dotenv import find_dotenv, load_dotenv
from data_utils import download_all
import preprocessing



def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('start downloading data into raw:')
    download_all()
    logger.info('downloading was successfull')
    # Not implemented yet
    preprocessing.preprocess()

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
