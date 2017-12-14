import pandas as pd
import numpy as np
from copy import deepcopy
import os
from data.download import DatasetDownloader
import tarfile
import sys
from scipy.interpolate import interp1d
from pyts.visualization import plot_paa
from pyts.transformation import PAA


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

        preprocessed_data = {}

        for token in tokens:
            # 1. Get travel data per token, remove dataframes without annotations.
            dfs = Preprocessor._remove_dataframes_without_annotation(
                Preprocessor.get_data_per_token(token)
            )

            # 2. Remove trips less than 10 minutes long.
            dfs = Preprocessor._remove_dataframes_by_duration_limit(dfs, 10 * 60)

            # 3. Cut first and last 30 seconds from scripted trips.
            dfs = Preprocessor._cut_off_start_and_end_in_dataframes(
                dataframes=dfs, list_of_dataframe_names_to_cut=["sensor", "location"], cutoff_in_seconds=30
            )

            # 4.Convert timestamps to human readable format
            #dfs = Preprocessor.convert_timestamps(dfs, unit="ms")

            # 5. Resample time series.
            #resampled_sensor_values = Preprocessor._resample_trip_time_series(dfs)
            resampled_sensor_values = Preprocessor.calculate_paa(dfs)

            #print(resampled_sensor_values[0]['sensor']['total'])

            # 6. Drop nan values:
            drop_almost_all_nans_ratio = 0.001
            #resampled_sensor_values = Preprocessor._filter_nan_values(resampled_sensor_values,  properties_to_check=["total"],
            #                                                          allowed_nan_ratio=drop_almost_all_nans_ratio)

            # 7. Recalculate 2-norm for accerelometer data.
            #resampled_sensor_values = Preprocessor._recalculate_accerelometer_2norm(resampled_sensor_values)


            # Prepare dictionary with results.
            preprocessed_data[token] = {}
            preprocessed_data[token]["trips"] = dfs
            preprocessed_data[token]["resampled_sensor_data"] = resampled_sensor_values

        return preprocessed_data

    @staticmethod
    def _filter_nan_values(dataframes: list, properties_to_check: list, allowed_nan_ratio: float = 0.2):
        """
        Filter NAN values from dataframes sensor data. Note that dataframe is dismissed if at least (!) one of the
        specified columns exceeds the allowed ratio of NANs.
        :param dataframes:
        :param properties_to_check: Properties to check for NAN values (e.g.: "sensor", "location").
        :param allowed_nan_ratio: Dataframe is removed if ratio (0 to 1) of NAN values relative to total count
        exceeds defined threshold.
        :return:
        """

        filtered_dataframes = []
        for i, df in enumerate(dataframes):
            # Check if threshold was reached for one of the specified columns.
            threshold_reached = True if np.count_nonzero(
                [
                    df[prop].isnull().sum().sum() / float(len(df[prop])) > allowed_nan_ratio
                    for prop in properties_to_check
                ]
            ) > 0 else False

            # Dismiss dataframe if share of NAN values is above defined_threshold.
            if not threshold_reached:
                # Drop rows with NANs.
                for key in properties_to_check:
                    df[key].dropna(axis=0, how='any', inplace=True)
                # Append to list.
                filtered_dataframes.append(df)

        return filtered_dataframes

    @staticmethod
    def _recalculate_accerelometer_2norm(resampled_dataframes):
        """
        Recalculates 2-norm for x-/y-/z-values in accerelometer data.
        Note that the original column 'total' is overwritten with the new value.
        :param resampled_dataframes:
        :return: List of dataframes with updated values for column 'total'.
        """

        for i, df in enumerate(resampled_dataframes):
            # Chain x-/y-/z-valuees for all entries in current dataframe and apply 2-norm on resulting (2, n)-shaped
            # vector.
            resampled_dataframes[i]["total"] = np.linalg.norm(
                np.array([df["x"], df["y"], df["z"]]),
                ord=2, axis=0
            )

        return resampled_dataframes

    @staticmethod
    def _cut_off_start_and_end_in_dataframes(dataframes, list_of_dataframe_names_to_cut, cutoff_in_seconds=30):
        """
        Auxiliary method with boilerplate code for cutting off start and end of timeseries in specified list of
        dataframes.
        :param dataframes:
        :param list_of_dataframe_names_to_cut:
        :param cutoff_in_seconds:
        :return: List of cleaned/cut dataframes.
        """

        scripted_trips = {"TRAM": 0, "METRO": 0, "WALK": 0}
        for i, df in enumerate(dataframes):
            # Assuming "notes" only has one entry per trip and scripted trips' notes contain the word "scripted",
            # while ordinary trips' notes don't.
            if "scripted" in str(df["annotation"]["notes"][0]).lower():
                scripted_trips[df["annotation"]["mode"][0]] += 1
                for dataframe_name in list_of_dataframe_names_to_cut:
                    # Cut off time series data.
                    dataframes[i][dataframe_name] = Preprocessor._cut_off_start_and_end_in_dataframe(
                        dataframe=df[dataframe_name], cutoff_in_seconds=cutoff_in_seconds
                    )
        print(scripted_trips)
        return dataframes

    @staticmethod
    def _cut_off_start_and_end_in_dataframe(dataframe, cutoff_in_seconds=30):
        """
        Removes entries with first and last cutoff_in_seconds in series.
        Assumes time in dataframe is specified in milliseconds.
        :param dataframe: Dataframe containing time series data. Expects specified dataframe to have a column "time".
        :param cutoff_in_seconds:
        :return: Cleaned dataframe.
        """

        # Only cut if enough values exist. If not (e. g. "location" data not available) return None.
        if not dataframe.empty:
            # Calculate time thresholds.
            lower_time_threshold = dataframe.head(1)["time"].values[0]
            upper_time_threshold = dataframe.tail(1)["time"].values[0]

            # Assuming time is specified as UTC timestamp in milliseconds, so let's convert the cutoff to milliseconds.
            cutoff_in_seconds *= 1000

            # Drop all rows with a time value less than 30 seconds after the initial entry and less than 30 seconds
            # before the last entry.
            dataframe = dataframe[
                (dataframe["time"] >= lower_time_threshold + cutoff_in_seconds) &
                (dataframe["time"] <= upper_time_threshold + cutoff_in_seconds)
            ]

            return dataframe

        else:
            return None

    @staticmethod
    def _resample_trip_time_series(dataframes):
        """
        Resamples trips' time series to hardcoded value (? per second).
        Returns list of dataframes with resampled time series.
        :param dataframes: List of dataframes with trip information.
        :return: List of resampled time series.
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
            if ("annotation" in df.keys()) and (not df["annotation"].empty):
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
        result = pd.DataFrame()
        if df is not None:
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
        copied_series = deepcopy(series)
        series_column_names = list(copied_series.columns.values)
        result = pd.DataFrame(columns = series_column_names)
        # In case the column or index has not been converted to Datetime object
        # it will be converted here.
        if (time_col_name=="index") and (copied_series.index.dtype in [np.dtype("Int64")]):

            copied_series.index = pd.to_datetime(copied_series.index, unit="ms")
        if time_col_name in series_column_names:
            if copied_series[time_col_name].dtype in [np.dtype("Int64")]:
                copied_series = Preprocessor._convert_timestamps_from_dataframe(copied_series, time_col_names=[time_col_name])

        # Start actual downsampling
        if isinstance(copied_series.index, pd.DatetimeIndex) or (time_col_name in series_column_names):
            for categorical_colname_i in categorical_colnames:
                categories = list(copied_series[categorical_colname_i].unique())
                for category_i in categories:
                    series_for_category = copied_series[copied_series[categorical_colname_i]==category_i]
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
            result = copied_series
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
        for index in range(0, nr_of_recorded_trips_token):
            trip_i = all_trips_copy[index]
            if ("annotation" in trip_i.keys()) and (not trip_i["annotation"].empty):
                result = pd.concat([result, trip_i["annotation"]])
                start_times.append(trip_i["marker"].iloc[0,0])
                end_times.append(trip_i["marker"].iloc[-1,0])

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

    @staticmethod
    def calculate_paa(dfs):
        # new dict
        newDict = deepcopy(dfs)
        for i in range(0, len(newDict)):
            print('Frame ', i)
            #get single trip
            sensor_trip = newDict[i]['sensor']
            #get all sensors
            sensor_set = set(sensor_trip['sensor'])
            #create new data frame
            helper = pd.DataFrame()

            for sensor in sensor_set:
                print("sensor: ", sensor)
                # make deep copy of data frame
                sensor_data = deepcopy(sensor_trip[sensor_trip['sensor'] == sensor])

                #print('init time frame')
                #print(Preprocessor.convert_timestamps(sensor_data.head(1)))
                #print(Preprocessor.convert_timestamps(sensor_data.tail(1)))

                sensor_data = sensor_data.drop(['sensor', 'total'], axis=1)
                sensor_data.reset_index(drop=True,inplace=True)
                sensor_data = Preprocessor.approx_sensor(sensor_data, 100)

                start_index = 0
                stop_index = 1
                end_of_df = len(sensor_data)

                buffer_helper = pd.DataFrame()
                filler = pd.DataFrame()

                #print("end_of_df:", end_of_df)

                while stop_index <= end_of_df:
                    if start_index + 30000 <= end_of_df:
                        stop_index = stop_index + 30000
                    else:
                        stop_index = end_of_df+1

                    buffer_helper = deepcopy(sensor_data.iloc[start_index:stop_index,:])
                    buffer_helper = Preprocessor.normalize_trip(buffer_helper)

                    #print('normalization start:', start_index)
                    #print('normalization stop:', stop_index)
                    #print(Preprocessor.convert_timestamps(buffer_helper.head(1))['time'])
                    #print(Preprocessor.convert_timestamps(buffer_helper.tail(1))['time'])
                    #print('************************')

                    filler = filler.append(buffer_helper)
                    start_index = stop_index

                filler['sensor'] = sensor
                filler['total'] = np.linalg.norm(np.array([filler['x'], filler['y'], filler['z']]),ord=2, axis=0)
                helper = pd.concat([helper,filler])

                #print("complete frame")
                #print(Preprocessor.convert_timestamps(helper.head(1))['time'])
                #print(Preprocessor.convert_timestamps(helper.tail(1))['time'])
                #print('----------------------------')
            #print(helper.head(1))
            newDict[i]['sensor'] = helper
        return Preprocessor.convert_timestamps(newDict)

    @staticmethod
    def approx_sensor(acc, hz=None, atd_ms=None):
        """
        This method interpolates the observations at equidistant time stamps.
        e.g. specifying hz=20 will result in a data frame containing 20 observaitons per second.

        Returns
        -------
        df : a pandas DataFrame containing the interpolated series
        """
        # interpolate to a common sampling rate
        #
        # acc ... data.table(time,x,y,z)
        # atd_ms ... approximated time difference in milliseconds, default value = 10

        if(hz is None and atd_ms is None):
            atd_ms = 10
        elif (hz is not None and atd_ms is None):
            atd_ms = 1000/hz
        elif (hz is not None and atd_ms is not None):
            print("hz is overruled with atd_ms")
        #print("len befor interpolation: ", len(acc['time']))
        #print(len(acc['time'])-1)
        new_time = np.arange(acc['time'][0], acc['time'][len(acc['time'])-1], atd_ms)
        #print("len after interpolation: ", len(new_time))
        f_ax = interp1d(acc['time'],acc['x'])
        ax = list(f_ax(new_time))
        f_ay = interp1d(acc['time'],acc['y'])
        ay = list(f_ay(new_time))
        f_az = interp1d(acc['time'],acc['z'])
        az = list(f_az(new_time))

        df = pd.DataFrame({
            'time':new_time,
            'x':ax,
            'y':ay,
            'z':az,
            'total': np.linalg.norm(
                np.array([ax, ay, az]),
                ord=2, axis=0
            )
        })
        #print("lin approx")
        #print(Preprocessor.convert_timestamps(df.head(1)))
        #print(Preprocessor.convert_timestamps(df.tail(1)))

        return df

    @staticmethod
    def normalize_trip(trip):
        """
        This method performs a Piecewise Aggregate Approximation of a trip.
        trip... a dataframe which should be used
        w_size... the bin size.

        REQUIREMENT: package 'future'

        Returns
        -------
        df : a pandas DataFrame containing the interpolated series
        """
        copy_dummy = deepcopy(trip)

        paa = PAA(window_size=5, output_size=None, overlapping=False)
        container = list()
        #print("before check")
        for label in copy_dummy.columns:
            #print("label: ", label)
            arr = np.array([copy_dummy[label]], dtype=np.float64)
            #print("pre_transform")
            transf = paa.transform(arr)
            #print("after transf")
            container.append(list(transf[0]))

        df = pd.DataFrame(container,trip.columns).T
        df['time'] = [int(i) for i in df['time']]
        return df

    @staticmethod
    def plot_paa(feature, w_size=20):
        plot_paa(feature, window_size=w_size,output_size=None,overlapping=False,marker='o')
