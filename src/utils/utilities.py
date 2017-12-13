import os
import argparse
from dotenv import find_dotenv, load_dotenv


def str2bool(v):
    """ Argparse utility function for extracting boolean arguments.
    Original code is from: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def check_for_nan_values(trips_per_token, verbose=False):
    """
    Returns true if a nan value is found.
    Set to verbose=True to see specific information
    """
    all_tokens =  [os.environ.get("KEY_RAPHAEL"),
                   os.environ.get("KEY_MORITZ"),
                   os.environ.get("KEY_LUKAS")]

    found_nan = False
    for token in all_tokens:
        if verbose:
            print("Token:", token)
        for i in range(len(trips_per_token[all_tokens[0]]["resampled_sensor_data"])):
            nr_of_nan_values = trips_per_token[all_tokens[0]]["resampled_sensor_data"][i].isnull().sum().sum()
            if verbose:
                print(i,nr_of_nan_values)
            if nr_of_nan_values > 0:
                found_nan = True
                if verbose:
                    print("length:", len(trips_per_token[all_tokens[0]]["resampled_sensor_data"][i]))

    return found_nan
