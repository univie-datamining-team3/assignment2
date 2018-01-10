"""
This script provides functions for feature engineering.
"""

import numpy as np
import pandas as pd


def feature_engineering(df: pd.DataFrame, trips):
    """
    Function that calls all feature engineering functions
    """
    longest_decreasing_patterns = calculate_maximum_break_length(trips,
                                                             threshold=3)

    longest_increasing_patterns = calculate_maximum_acceleration_length(trips,
                                                             threshold=3)

    features = get_summary_features(df)
    features = features.drop(["count","min","max","mean"],axis=1)

    features["longest_break_period"] = np.array(longest_decreasing_patterns,dtype=np.float32)
    features["longest_acceleration_period"] = np.array(longest_increasing_patterns,dtype=np.float32)

    return features


def async (trips: list):
    """
    Uses tsfresh (https://github.com/blue-yonder/tsfresh) to engineer features from time series.
    :param trips: List of dataframes for trips.
    :return:
    """

    print(trips.head(3))


def _get_longest_increasing_pattern(trips, threshold=5):
    """
    This function calculates the longest increasing pattern from a matrix
    of trips.
    """
    # make sure trips are numpy array
    trips_copy = np.array(trips, dtype=np.float32)
    longest_increasing_patterns = []
    nr_of_trips = trips_copy.shape[0]

    for index in range(nr_of_trips):
        trip_i = np.array(trips_copy[index,:])
        larger_values = []
        larger_value = trip_i[0]
        nr_of_consecutive_larger_values = []
        found_smaller_value = 0
        helper_list = []
        for value_i in trip_i:
            if larger_value <= value_i:
                larger_value = value_i
                larger_values.append(larger_value)
            else:
                found_smaller_value += 1
                helper_list.append(len(larger_values))
            if found_smaller_value > threshold:
                nr_of_consecutive_larger_values.append(sum(helper_list))
                larger_value = 0.0
                found_smaller_value = 0
                larger_values = []
                helper_list = []

        # Test for edge case that no larger value was found
        if not nr_of_consecutive_larger_values:
            if not helper_list:
                longest_increasing_patterns.append(0)
            else:
                longest_increasing_patterns.append(np.max(helper_list))
        else:
            longest_increasing_patterns.append(np.max(nr_of_consecutive_larger_values))

    return longest_increasing_patterns

def _get_longest_decreasing_pattern(trips, threshold=5):
    trips_copy = np.array(trips, dtype=np.float32) * -1.0
    return _get_longest_increasing_pattern(trips_copy, threshold=threshold)


def calculate_maximum_break_length(trips, threshold=5, time_step=1/20.0):
    """
    Calculate the maximum break length for each trip. The naive assumption
    is that we just have to find the longest decreasing series of numbers for
    each trip. E.g. if we have [5,4,3,2,3,6] the longest decreasing pattern is
    5,4,3,2 with length 4. The threshold value indicates how many values could
    interrupt the pattern before it is invalid, this is based on the observation,
    that most trips have outliers during the breaking period.

    Parameters
    ----------
    trips: a numpy.array or pandas.DataFrame where each row is a trip
    threshold: int, default=5
        specifies how many values are allowed to interrupt the pattern

    Returns
    -------
    a list of one value per trip, which indicates the maximum break length in recording
    steps. Note that this is only counting values and not measuring seconds. In
    our case we have to multiply each value by 1/20 seconds.
    """
    ts = float(time_step)
    return [i*ts for i in _get_longest_decreasing_pattern(trips, threshold=threshold)]


def calculate_maximum_acceleration_length(trips, threshold=5, time_step=1/20.0):
    """
    Calculate the maximum acceleration length for each trip. The naive assumption
    is that we just have to find the longest increasing series of numbers for
    each trip. E.g. if we have [1,2,3,4,2,1] the longest increasing pattern is
    1,2,3,4 with length 4. The threshold value indicates how many values could
    interrupt the pattern before it is invalid, this is based on the observation,
    that most trips have outliers during the acceleration period.

    Parameters
    ----------
    trips: a numpy.array or pandas.DataFrame where each row is a trip
    threshold: int, default=5
        specifies how many values are allowed to interrupt the pattern
    time_step: float, default=1/20.0,
        time step for each recording, in our case 1/20, due to resampling of hertz rate
        to 20.

    Returns
    -------
    a list of one value per trip, which indicates the maximum acceleration length in recording
    steps. Note that this is only counting values and not measuring seconds. In
    our case we have to multiply each value by 1/20 seconds.
    """

    ts = float(time_step)
    return [i*ts for i in _get_longest_increasing_pattern(trips, threshold=threshold)]

def get_summary_features(df: pd.DataFrame, percentiles=None):
    if percentiles is None:
        percentiles=[0.1*(i+1) for i in range(10)]

    summary_features = df.transpose().describe(percentiles=percentiles).T

    return summary_features
