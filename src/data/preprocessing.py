"""
File for preprocessing routines on the mobility data set
"""
import pandas as pd
import numpy as np
from copy import deepcopy

def preprocess():
    # TODO
    pass



def _convert_timestamps_from_dataframe(df, unit="ms", time_col_names=["time","gpstime"]):
    """
    This method converts integer timestamp columns in a pandas.DataFrame object
    to datetime objects.
    DataFrames in mobility data with datetime columns:
    cell, event, location, marker, sensor

    Parameters
    ----------
    data: input data pandas DataFrame.
    unit: string, default="ms"
          time unit for transformation of the input integer timestamps.
          Possible values: D,s,ms,us,ns
          see "unit" at: http://pandas.pydata.org/pandas-docs/stable/generated/pandas.to_datetime.html
          for further information.

    Returns
    -------
    data: returns a deepcopy of the data with transformed time columns.
    """

    df_column_names = list(df.columns.values)
    if any(name_i in time_col_names for name_i in df_column_names):
        result = deepcopy(df)
        for time_column_name in time_col_names:
            if time_column_name in df_column_names:
                index_copy = result.index
                result.set_index(time_column_name,inplace=True)
                result.index = pd.to_datetime(result.index, unit=unit)
                result.reset_index(inplace=True)
                result.index = index_copy
    return result

def _convert_timestamps_from_dictionary_of_dataframes(d, unit="ms", time_col_names=["time","gpstime"]):
    """ Convenience function to loop over dicionary of one track recording.
    """
    result = dict()
    for df_name, df in d.items():
        result[df_name] = _convert_timestamps_from_dataframe(df,unit=unit, time_col_names=time_col_names)

    return result

def _convert_timestamps_from_list_of_total_trips(all_trips, unit="ms", time_col_names=["time","gpstime"]):
    """ Convenience function to loop over list af all track recordings.
    """
    for i, trip_i in enumerate(all_trips):
        all_trips[i] = _convert_timestamps_from_dictionary_of_dataframes(trip_i, unit=unit, time_col_names=time_col_names)
    return all_trips

def convert_timestamps(data, unit="ms", time_col_names=["time","gpstime"]):
    """
    This function converts the integer timestamps in the specified columns to
    datetime objects in the format YYYY-MM-DD HH-MM-SS-uu-.., where uu stands for
    the specified unit.
    It is assumed that the time colums are integer as it is the case for the mobility data.
    Accepted input types are pandas.DataFrame, dict, list which follow the convention
    of the projects nesting structure. Special case if data is of type pandas.DataFrame
    then the behaviour of this function equals _convert_timestamps_from_dataframe:

    Parameters
    ----------
    data: input data, can be a list of all tracks, a dict of one track or a
          pandas DataFrame of one table.
    unit: string, default="ms"
          time unit for transformation of the input integer timestamps.
          Possible values: D,s,ms,us,ns
          see "unit" at: http://pandas.pydata.org/pandas-docs/stable/generated/pandas.to_datetime.html
          for further information.
    time_col_names: list of strings, default=["time","gpstime"]
        names of the time colums in the table which should be transformed.
    Returns
    -------
    data: returns a deepcopy of the data with transformed time columns.
          The datatype of data will be the same as of the input type. Accepted input types are
          pandas.DataFrame, dict, list.

    """
    if type(data) is pd.DataFrame:
        data = _convert_timestamps_from_dataframe(data, unit, time_col_names)
    elif type(data) is dict:
        data = _convert_timestamps_from_dictionary_of_dataframes(data, unit, time_col_names)
    elif type(data) is list:
        data = _convert_timestamps_from_list_of_total_trips(data, unit, time_col_names)

    return data



def downsample_time_series(series, time_interval="S", time_col_name="time"):
    """
    Downsamples a pandas time series DataFrame from milliseconds to a new
    user specified time interval. The aggregation for the new time bins will be
    calculated via the mean. To make sure that the right time column is
    used you have to set the time columns name in time_col_name or set it as
    index before calling this function.
    Otherwise it is assumed that the time column has the name time_col_name="time".

    For further information about examples for pandas resampling function see:
    http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.resample.html
    http://pandas.pydata.org/pandas-docs/stable/timeseries.html#resampling
    https://machinelearningmastery.com/resample-interpolate-time-series-data-python/


    Parameters
    ----------
    series: a pandas DataFrame object with a DatetimeIndex, if there is no
        DatetimeIndex set, it is assumed that there is a Datetime column with
        name time_col_name="time"
    time_interval: string, default="S",
        specifies the new time interval to which the series will be downsampled.
        Valid values are "S" for seconds, "T" for minutes etc. It is also possible
        to sample in a special interval e.g. 5 seconds, by passing "5S".
        For all possible frequencies see:
        https://stackoverflow.com/questions/17001389/pandas-resample-documentation#17001474

    time_col_name: string, default="time"
        The name of the time column name.
    Returns
    -------
    data: returns the data with downsampled time columns, where each new bin
          is aggregated via the mean.
    """

    if isinstance(series.index, pd.DatetimeIndex):
        resampled = series.resample(time_interval).mean()
    elif time_col_name in list(series.columns.values):
        # In case the column has not been converted to Datetime object
        # it will be converted here.
        if series[time_col_name].dtype in [np.dtype("Int64")]:
            series = deepcopy(series)
            series = convert_timestamps(series, time_col_names=[time_col_name])
        resampled = series.set_index(time_col_name).resample(time_interval).mean()
        resampled = resampled.reset_index()
    else:
        resampled = series
    return resampled

def downsample_time_series_per_category(series, categorical_colnames, time_interval="S", time_col_name="time"):
    """
    Downsamples a pandas time series DataFrame from milliseconds to a new
    user specified time interval and takes care of the right interpolation of categorical variables.
    The aggregation for the new time bins will be calculated via the mean.
    To make sure that the right time column is used you have to set the time
    columns name in time_col_name.
    Otherwise it is assumed that the time column has the name time_col_name="time".

    For further information about examples for pandas resampling function see:
    http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.resample.html
    http://pandas.pydata.org/pandas-docs/stable/timeseries.html#resampling
    https://machinelearningmastery.com/resample-interpolate-time-series-data-python/


    Parameters
    ----------
    series: a pandas DataFrame object with a DatetimeIndex, if there is no
        DatetimeIndex set, it is assumed that there is a Datetime column with
        name time_col_name="time"
    categorical_colnames: a list of strings of colum names
        e.g. ["sensor"]
    time_interval: string, default="S",
        specifies the new time interval to which the series will be downsampled.
        Valid values are "S" for seconds, "T" for minutes etc. It is also possible
        to sample in a special interval e.g. 5 seconds, by passing "5S".
        For all possible frequencies see:
        https://stackoverflow.com/questions/17001389/pandas-resample-documentation#17001474
    time_col_name: string, default="time"
        The name of the time column name. set to "index" if you want to transform
        the index column
    Returns
    -------
    data: returns the data with downsampled time columns, where each new bin
          is aggregated via the mean and keeps the categorical values.
    """
    series_column_names = list(series.columns.values)
    result = pd.DataFrame(columns = series_column_names)
    # In case the column or index has not been converted to Datetime object
    # it will be converted here.
    if (time_col_name=="index") and (series.index.dtype in [np.dtype("Int64")]):
        series = deepcopy(series)
        series.index = pd.to_datetime(series.index, unit="ms")
    if time_col_name in series_column_names:
        if series[time_col_name].dtype in [np.dtype("Int64")]:
            series = deepcopy(series)
            series = _convert_timestamps_from_dataframe(series, time_col_names=[time_col_name])

    # Start actual downsampling
    if isinstance(series.index, pd.DatetimeIndex) or (time_col_name in series_column_names):
        for categorical_colname_i in categorical_colnames:
            categories = list(series[categorical_colname_i].unique())
            for category_i in categories:
                series_for_category = series[series[categorical_colname_i]==category_i]
                resampled = downsample_time_series(series_for_category, time_interval, time_col_name)
                resampled[categorical_colname_i] = category_i
                result = pd.concat([result, resampled])

        if isinstance(result.index, pd.DatetimeIndex):
            result = result.sort_index()
        else:
            result = result.set_index(time_col_name).sort_index()
        # need to reset index otherwise indices could be not unique anymore
        result = result.reset_index()
    else:
        result = series
    return result
