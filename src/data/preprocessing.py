import pandas as pd
import numpy as np
from copy import deepcopy
import os
from data.download import DatasetDownloader
import tarfile
import sys


class Preprocessor:
    """
    Class for preprocessing routines on the mobility data set.
    """

    @staticmethod
    def preprocess(tokens):
        """
        Executes all preprocessing steps.
        :param tokens: List with keys of tokens to preprocess.
        :return: Dictionary with preprocessed data. Specified tokens are used as keys.
        """
        pd.set_option('display.width', 1000)
        pd.set_option('display.max_columns', 20)

        preprocessed_data = {}

        for token in tokens:
            # 1. Get travel data per token, remove dataframes without annotations.
            dfs = Preprocessor._remove_dataframes_without_annotation(
                Preprocessor.get_data_per_token(tokens[0])
            )

            # 2. Remove trips less than 10 minutes long.
            dfs = Preprocessor._remove_dataframes_by_duration_limit(dfs, 10 * 60)

            # 3. Resample time series.
            resampled_sensor_values = Preprocessor._resample_trip_time_series(dfs)

            preprocessed_data[token] = {}
            preprocessed_data["dataframes"] = dfs
            preprocessed_data["resampled_sensor_data"] = resampled_sensor_values

        return preprocessed_data

    @staticmethod
    def _resample_trip_time_series(dataframes):
        """
        Resamples trips' time series to hardcoded value (? per second).
        Returns list of dataframes with resampled time series.
        :param dataframes: List of dataframes with trip information.
        :return:
        """

        return [
            Preprocessor.downsample_time_series_per_category(df["sensor"], categorical_colnames=["sensor"])
            for df in dataframes
        ]

    @staticmethod
    def _remove_dataframes_without_annotation(dataframes):
        """
        Removes dataframes w/o annotation data (since we don't know the transport mode and hence can't use it for
        training.
        :param dataframes: ist of dataframes with trip data.
        :return:
        """

        filtered_dataframes = []
        for df in dataframes:
            if not df["annotation"].empty:
                filtered_dataframes.append(df)

        return filtered_dataframes

    @staticmethod
    def _remove_dataframes_by_duration_limit(dataframes, min_duration=0, max_duration=sys.maxsize):
        """
        Removes dataframes outside the defined time thresholds.
        :param dataframes: ist of dataframes with trip data.
        :param min_duration: Minimum duration in seconds.
        :param max_duration: Maximum duration in seconds.
        :return:
        """

        # Fetch summaries for all trips.
        trip_summaries = Preprocessor.get_trip_summaries(dataframes, convert_time=True)

        filtered_dataframes = []
        for i, df in enumerate(dataframes):
            trip_length = trip_summaries.iloc[i]["trip_length"].total_seconds()
            if trip_length >= min_duration and trip_length >= min_duration:
                filtered_dataframes.append(df)

        return filtered_dataframes

    @staticmethod
    def _convert_timestamps_from_dataframe(df, unit="ms", time_col_names=["time", "gpstime"]):
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
        result: returns a deepcopy of the data with transformed time columns.
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

    @staticmethod
    def _convert_timestamps_from_dictionary_of_dataframes(d, unit="ms", time_col_names=["time","gpstime"]):
        """ Convenience function to loop over dicionary of one track recording.
        """
        result = dict()
        for df_name, df in d.items():
            result[df_name] = Preprocessor._convert_timestamps_from_dataframe(df,unit=unit, time_col_names=time_col_names)
        return result

    @staticmethod
    def _convert_timestamps_from_list_of_total_trips(all_trips, unit="ms", time_col_names=["time","gpstime"]):
        """ Convenience function to loop over list af all track recordings.
        """
        result = []
        for i, trip_i in enumerate(all_trips):
            result.append(Preprocessor._convert_timestamps_from_dictionary_of_dataframes(trip_i, unit=unit, time_col_names=time_col_names))
        return result

    @staticmethod
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
        result: returns a deepcopy of the data with transformed time columns.
              The datatype of data will be the same as of the input type. Accepted input types are
              pandas.DataFrame, dict, list.

        """
        result = pd.DataFrame()
        if type(data) is pd.DataFrame:
            result = Preprocessor._convert_timestamps_from_dataframe(data, unit, time_col_names)
        elif type(data) is dict:
            result = Preprocessor._convert_timestamps_from_dictionary_of_dataframes(data, unit, time_col_names)
        elif type(data) is list:
            result = Preprocessor._convert_timestamps_from_list_of_total_trips(data, unit, time_col_names)

        return result

    @staticmethod
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
                series = Preprocessor.convert_timestamps(series, time_col_names=[time_col_name])
            resampled = series.set_index(time_col_name).resample(time_interval).mean()
            resampled = resampled.reset_index()
        else:
            resampled = series
        return resampled

    @staticmethod
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
                series = Preprocessor._convert_timestamps_from_dataframe(series, time_col_names=[time_col_name])

        # Start actual downsampling
        if isinstance(series.index, pd.DatetimeIndex) or (time_col_name in series_column_names):
            for categorical_colname_i in categorical_colnames:
                categories = list(series[categorical_colname_i].unique())
                for category_i in categories:
                    series_for_category = series[series[categorical_colname_i]==category_i]
                    resampled = Preprocessor.downsample_time_series(series_for_category, time_interval, time_col_name)
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

    @staticmethod
    def get_trip_summaries(all_trips, convert_time=False):
        """
        This method returns a summary of all recorded trips. The summary includes start,
        stop time, trip_length, recording mode and notes.

        Parameters
        ----------
        all_trips : a list of all trips
        convert_time : bool, default=False
            indicates whether or not the time values should be converted to datetime
            objects.

        Returns
        -------
        result : pandas DataFrame
            a pandas dataframe with the summaries for each trip
        """
        nr_of_recorded_trips_token = len(all_trips)
        result = pd.DataFrame()
        if convert_time:
            all_trips_copy = Preprocessor.convert_timestamps(all_trips)
        else:
            all_trips_copy = all_trips
        start_times = []
        end_times = []
        for trip_i in range(0, nr_of_recorded_trips_token):
            result = pd.concat([result, all_trips_copy[trip_i]["annotation"]])
            start_times.append(all_trips_copy[trip_i]["marker"].iloc[0,0])
            end_times.append(all_trips_copy[trip_i]["marker"].iloc[-1,0])

        result["Start"] = start_times
        result["Stop"] = end_times
        result["trip_length"] = [end-start for end,start in zip(end_times,start_times)]
        result = result.reset_index(drop=True)

        return result

    @staticmethod
    def extract_csv_file_name(csv_name):
        """
        Extracts the name from the csv file name e.g. annotation, cell, event, location,
        mac, marker, sensor.

        Parameters
        ----------
        csv_name: full name of the csv file in tar.gz directory
        Returns
        -------
        extracted_name: string,
        """
        csv_name = str(csv_name)
        extracted_name = ""
        for name in DatasetDownloader.VALID_NAMES:
            if name in csv_name:
                extracted_name = name
                return extracted_name

        return extracted_name

    @staticmethod
    def read_tar_file_from_dir(file_path):
        """
        This method reads a tar.gz file from a specified file path and appends each
        .csv file to a dictionary where the key is specified as one of the VALID_NAMES:
        ["annotation", "cell", "event", "location", "mac", "marker", "sensor"], which
        are the names given to identify the different collected mobility data.

        """
        tar = tarfile.open(file_path, "r:gz")
        csv_files_per_name = {}
        for member in tar.getmembers():
            f = tar.extractfile(member)
            if f is not None:
                name = Preprocessor.extract_csv_file_name(member)
                csv_files_per_name[name] = pd.read_csv(f, header=0, sep=',', quotechar='"')
        tar.close()
        return csv_files_per_name

    @staticmethod
    def get_data_per_trip(dir_name="raw"):
        """
        This method reads all downloaded data and returns a list of dictionaries
        which include the pandas dataframes for each trip. Each trip DataFrame
        can be accessed via its name e.g. annotation, cell, event, location,
        mac, marker, sensor.

        Parameters
        -------
        dir_name : string, default="raw",
            specifies the name of the directory inside the data directory from which
            the data should be read.


        Returns
        -------
        data_frames : a list of  pandas DataFrame's in a dictionary
        """

        file_path = os.path.join(Preprocessor.get_data_dir(), dir_name)
        tar_file_names = Preprocessor.get_file_names(file_path)
        dfs = []
        for tar_name in tar_file_names:
            path_to_tar_file = os.path.join(file_path, tar_name)
            csv_files_per_name = Preprocessor.read_tar_file_from_dir(path_to_tar_file)
            dfs.append(csv_files_per_name)
        return dfs

    @staticmethod
    def get_data_per_token(token):
        """
        This method reads the downloaded data for one user and returns a list of dictionaries
        which include the pandas dataframes for each trip. Each trip DataFrame
        can be accessed via its name e.g. annotation, cell, event, location,
        mac, marker, sensor.

        Returns
        -------
        data_frames : a list of  pandas DataFrame's in a dictionary
        """
        file_path = os.path.join(DatasetDownloader.get_data_dir(), "raw")
        tar_file_names = DatasetDownloader.get_file_names_for(file_path, token)
        dfs = []
        for tar_name in tar_file_names:
            path_to_tar_file = os.path.join(file_path, tar_name)
            csv_files_per_name = Preprocessor.read_tar_file_from_dir(path_to_tar_file)
            dfs.append(csv_files_per_name)

        return dfs
