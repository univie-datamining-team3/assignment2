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
import pickle
from scipy.spatial.distance import cdist, squareform
from data.DTWThread import DTWThread
import psutil


class Preprocessor:
    """
    Class for preprocessing routines on the mobility data set.
    """

    # Names of columns in all dataframes. Used to inject columns into empty dataframes.
    DATAFRAME_COLUMN_NAMES = {
        "cell": ['time', 'cid', 'lac', 'asu'],
        "annotation": ['time', 'mode', 'notes'],
        "location": ['time', 'gpstime', 'provider', 'longitude', 'latitude', 'altitude', 'speed', 'bearing',
                     'accuracy'],
        "sensor": ['sensor', 'time', 'x', 'y', 'z', 'total'],
        "mac": ['time', 'ssid', 'level'],
        "marker": ['time', 'marker'],
        "event": ['time', 'event', 'state']
    }

    @staticmethod
    def preprocess(tokens, filename: str = None, distance_metric: str = "euclidean", use_individual_columns: bool = False, load_preprocessed: str = None):
        """
        Executes all preprocessing steps.
        :param tokens: List with keys of tokens to preprocess.
        :param filename: Specifies name of file data should be dumped to. Not persisted to disk if specified value is
        None. Note that filename is relative; all files are stored in /data/preprocessed.
        :param distance_metric: Distance metric to apply for comparison between trip segments.
        :param use_individual_columns: Defines whether individual columns (x, y, z) or the total (n2) value should be
        used for distance calculation.
        :load_preprocessed: str, default=None, specifies a path to a pickled preprocessed_data.dat file.
            if this parameter is not None the preprocessing step is skipped and the pickled data will be
            loaded.
        :return: Dictionary with preprocessed data. Specified tokens are used as keys.
        """

        # 1. Preprocess data per token.
        if load_preprocessed is not None:
            # Load dataframes from disk.
            preprocessed_data = Preprocessor.restore_preprocessed_data_from_disk(filename=load_preprocessed)
        else:
            preprocessed_data = Preprocessor._preprocess_data_per_token(tokens=tokens)

        # 2. Cut all trips in 30 second snippets
        trips_cut_per_30_sec = Preprocessor.get_cut_trip_snippets_for_targets(
            preprocessed_data,
            snippet_length=30,
            sensor_type="acceleration",
            target_column_names=["total", "x", "y", "z"]
        )

        print("dimensions:", trips_cut_per_30_sec[0].shape)

        # 3. Apply distance metric and calculate distance matrix
        # distance_matrix = None
        # if distance_metric is not None:
        #     if use_individual_columns:
        #         distance_matrix = Preprocessor.calculate_distance_for_individual_columns(
        #                 dataframes=trips_cut_per_30_sec[1:4]
        #         )
        #     else:
        #         distance_matrix = Preprocessor.calculate_distance_for_n2(
        #             trips_cut_per_30_sec[0],
        #             metric=distance_metric
        #         )

        # 4. Dump data to file, if requested.
        if filename is not None:
            Preprocessor.persist_results(
                filename=filename,
                preprocessed_data=preprocessed_data,
                trips_cut_per_30_sec=trips_cut_per_30_sec,
                distance_metric=distance_metric,
                distance_matrix_n2=distance_matrix
                use_individual_columns=use_individual_columns
            )

    @staticmethod
    def _preprocess_data_per_token(tokens: list):
        """
        List of tokens whose data is to be processed.
        :param tokens:
        :return: Dictionary with preprocessed data per token.
        """

        preprocessed_data = {}

        for token in tokens:
            # 1. Get travel data per token, remove dataframes without annotations.
            dfs = Preprocessor.replace_none_values_with_empty_dataframes(
                # Drop dataframes w/o annotations.
                Preprocessor._remove_dataframes_without_annotation(
                    # Get travel data per token.
                    Preprocessor.get_data_per_token(token)
                )
            )

            # 2. Remove trips less than 10 minutes long.
            dfs = Preprocessor.replace_none_values_with_empty_dataframes(
                Preprocessor._remove_dataframes_by_duration_limit(dfs, 10 * 60)
            )

            # 3. Cut first and last 30 seconds from scripted trips.
            dfs = Preprocessor.replace_none_values_with_empty_dataframes(
                Preprocessor._cut_off_start_and_end_in_dataframes(
                    dataframes=dfs, list_of_dataframe_names_to_cut=["sensor", "location"], cutoff_in_seconds=60
                )
            )

            # 4. Perform PAA.
            resampled_sensor_values = Preprocessor.replace_none_values_with_empty_dataframes(
                Preprocessor.calculate_paa(dfs)
            )

            # Prepare dictionary with results.
            preprocessed_data[token] = resampled_sensor_values

        return preprocessed_data

    @staticmethod
    def persist_results(filename: str, preprocessed_data: dict, trips_cut_per_30_sec: list,
                        distance_metric: str, distance_matrix_n2: pd.DataFrame, use_individual_columns=False):
        """
        Stores preprocessing results on disk.
        :param filename:
        :param preprocessed_data:
        :param trips_cut_per_30_sec:
        :param distance_metric:
        :param distance_matrix_n2:
        :param use_individual_columns: indicates if individual columns were used
        :return:
        """

        data_dir = DatasetDownloader.get_data_dir()
        preprocessed_path = os.path.join(data_dir, "preprocessed")
        # make sure the directory exists
        DatasetDownloader.setup_directory(preprocessed_path)
        full_path = os.path.join(preprocessed_path, filename)
        with open(full_path, "wb") as file:
            file.write(pickle.dumps(preprocessed_data))

        trips_cut_per_30_sec[0].to_csv(full_path[:-4] + "_total.csv", sep=";", index=False)
        trips_cut_per_30_sec[1].to_csv(full_path[:-4] + "_x.csv", sep=";", index=False)
        trips_cut_per_30_sec[2].to_csv(full_path[:-4] + "_y.csv", sep=";", index=False)
        trips_cut_per_30_sec[3].to_csv(full_path[:-4] + "_z.csv", sep=";", index=False)

        if distance_metric is not None:
            if use_individual_columns:
                distance_matrix_n2_path = full_path[:-4] + "_" + "individual" + "_" + distance_metric + "_xyz" +".csv"
            else:
                distance_matrix_n2_path = full_path[:-4] + "_" + distance_metric + ".csv"
            distance_matrix_n2.to_csv(distance_matrix_n2_path, sep=";", index=False)

    @staticmethod
    def replace_none_values_with_empty_dataframes(dataframe_dicts: list):
        """
        Checks every dictionary in every dictionary in specified list for None values, replaces them with empty data-
        frames.
        :param dataframe_dicts: List of dictionaries containing one dataframe for each key.
        :return: List in same format with Nones replaced by empty dataframes.
        """

        # For every key in every dictionary in list: Create new dictionary with Nones replaced by empty dataframes;
        # concatenate new dictionaries to list.
        return [
            {
                key: pd.DataFrame(columns=Preprocessor.DATAFRAME_COLUMN_NAMES[key])
                if df_dict[key] is None else df_dict[key]
                for key in df_dict
            } for df_dict in dataframe_dicts
        ]

    @staticmethod
    def get_cut_trip_snippets_for_targets(dfs, target_column_names: list, snippet_length=30, sensor_type="acceleration"):
        """
        This method gets a dictionary of trips per token and cuts them in the
        specified snippet_length. It uses the columns of the specified names
        (i. e. one or several of: "total", "x", "y", "z") in the sensor table.

        Parameters
        ----------
        dfs: dictionary with the assumed nested structure
            dict[token] = list of trips per token and each trip consists of tables for
            at least "annotation" and "sensor"
        snippet_length: int, default=30,
            specifies the length of the time snippets in seconds
        sensor_type: string, default="acceleration"
            specifies which sensor type should be used for each entry
        target_column_names: list
            Specifies which columns should represent trip observation.

        Returns
        -------
        result: returns a list of pandas.DataFrames where each row is a snippet with length snippet_length
                and each column is one recording step. Each entry corresponds
                to the total aka n2 value of the original data. Additional columns are:
                "mode","notes","scripted","token","trip_id", where scripted is a binary variable
                where scripted=1 and ordinary=0. "trip_id" helps to identify which snippet, belongs
                to which trip.
                Each element in the list corresponds to one of the specified target columns (in the same sequence).
        """

        return [
            Preprocessor.get_cut_trip_snippets_for_target(
                dfs=dfs,
                snippet_length=snippet_length,
                sensor_type=sensor_type,
                target_column_name=target_column
            )
            for target_column in target_column_names
        ]

    @staticmethod
    def get_cut_trip_snippets_for_target(dfs, snippet_length=30, sensor_type="acceleration", target_column_name: str = "total"):
        """
        This method gets a dictionary of trips per token and cuts them in the
        specified snippet_length. It uses the one dimensional column of the specified name
        (i. e. one of: "total", "x", "y", "z") in the sensor table.

        Parameters
        ----------
        dfs: dictionary with the assumed nested structure
            dict[token] = list of trips per token and each trip consists of tables for
            at least "annotation" and "sensor"
        snippet_length: int, default=30,
            specifies the length of the time snippets in seconds
        sensor_type: string, default="acceleration"
            specifies which sensor type should be used for each entry
        target_column_name: string, default="total"
                    Specifies which column should represent trip observation.

        Returns
        -------
        result: returns a pandas.DataFrame where each row is a snippet with length snippet_length
                and each column is one recording step. Each entry corresponds
                to the total aka n2 value of the original data. Additional columns are:
                "mode","notes","scripted","token","trip_id", where scripted is a binary variable
                where scripted=1 and ordinary=0. "trip_id" helps to identify which snippet, belongs
                to which trip.
        """
        HERTZ_RATE = 20
        column_names = ["snippet_"+str(i) for i in range(snippet_length * HERTZ_RATE)]
        column_names = column_names + ["mode","notes","scripted","token", "trip_id"]

        result = pd.DataFrame(columns=column_names)
        trip_index = 0
        for token_i, trips in sorted(dfs.items()):
            for trip_i in trips:
                sensor_data, mode, notes, scripted = Preprocessor._get_row_entries_for_trip(trip_i, sensor_type=sensor_type)
                splitted_trip = Preprocessor._cut_trip(
                    sensor_data=sensor_data,
                    target_column_name=target_column_name,
                    snippet_length=snippet_length,
                    column_names=column_names
                )
                splitted_trip["mode"]=mode
                if str(notes).lower() == "nan":
                    splitted_trip["notes"]="empty"
                else:
                    splitted_trip["notes"]=notes
                splitted_trip["scripted"]=scripted
                splitted_trip["token"]=token_i
                splitted_trip["trip_id"]=trip_index
                trip_index += 1
                result = pd.concat([result, splitted_trip])

        result.reset_index(drop=True, inplace=True)

        return result

    @staticmethod
    def calculate_distance_for_n2(data, metric="euclidean"):
        """
        This method calculates the specified distance metric for norms of the x,y,z signal,
        also called n2 or total in the assignment.

        Parameters
        ----------
        data: pandas.DataFrame of the trip segments and the
              ["mode","notes","scripted","token", "trip_id"] columns
        metric: string, default="euclidean",
            specifies which distance metric method should be used. The distance is calculated
            with the highly optimized cdist function of scipy.spatial.distance.
            This makes it simple to use a wide variety of distance metrics, some
            of them listed below.
            Mandatory Distance Calculations:
                "euclidean" : calculates the euclidean distance
                "cityblock" : calculates the manhattan distance
                "cosine"    : calculates the cosine distance
                "dtw"       : Calculates distance with dynamic time warping. Utilizes l1 norm.
            for a full list of all distances see:
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html


        Returns
        -------
        result: returns a pandas.DataFrame where each each point in the distance matrix
                is the distance of one trip segment to another one and each row of the
                distance matrix corresponds to the trips segment distances to all other
                trip segments.  Additional columns are: "mode","notes","scripted","token",
                where scripted is a binary variable where scripted=1 and ordinary=0
                Note that the dimensionality of the result can be (for most cases)
                different to the dimensionality of the incoming data pandas.DataFrame.

        """
        categorical_colnames=["mode","notes","scripted","token", "trip_id"]
        small_df = data.drop(categorical_colnames, axis=1)

        #column_names = list(small_df.columns.values)
        nr_of_rows =  small_df.shape[0]
        nr_of_columns = small_df.shape[1]
        # The new dataframe has dimensionality of nr_of_rows x nr_of_rows
        column_names = ["distance_"+str(i) for i in range(nr_of_rows)]
        result = pd.DataFrame(columns=column_names)

        distance_matrix = \
            cdist(small_df, small_df, metric=metric) if metric != 'dtw' \
            else Preprocessor._calculate_distance_with_dtw(small_df, 1)

        result = pd.concat([result, pd.DataFrame(distance_matrix, columns=column_names)])

        # Reappend the categorical columns
        for colname in categorical_colnames:
            result[colname] = data[colname]
        return result

    @staticmethod
    def calculate_distance_for_individual_columns(dataframes: list):
        """
        This method calculates the specified distance metric for the individual x, y, z columns.
        Note that due to the data structure required for calculating distances between the individual columns currently
        only the Euclidean norm is supported, since I haven't found a way to concile scipy's cdist-function with the
        additional dimension (individual columns) in the dataset.

        Parameters
        ----------
        dataframes: List of pandas.DataFrame of the trip segments and the
              ["mode","notes","scripted","token", "trip_id"] columns with length 3 - has to contain dataframe
              for columns "x", "y" and "z".
        Returns
        -------
        result: returns a pandas.DataFrame where each each point in the distance matrix
                is the distance of one trip segment to another one and each row of the
                distance matrix corresponds to the trips segment distances to all other
                trip segments.  Additional columns are: "mode","notes","scripted","token",
                where scripted is a binary variable where scripted=1 and ordinary=0
                Note that the dimensionality of the result can be (for most cases)
                different to the dimensionality of the incoming data pandas.DataFrame.

        """
        categorical_colnames=["mode","notes","scripted","token", "trip_id"]

        # Drop categorical column names for all dataframes.
        small_dfs = [data.drop(categorical_colnames, axis=1) for data in dataframes]

        # The new dataframe has dimensionality of nr_of_rows x nr_of_rows
        nr_of_rows = small_dfs[0].shape[0]
        column_names = ["distance_" + str(i) for i in range(nr_of_rows)]
        result = pd.DataFrame(columns=column_names)

        # Calculating distance matrix manually, since cdist(...) doesn't take 3D-arrays and I don't know how to solve
        # this more elegantly.
        distance_matrix = np.zeros([nr_of_rows, nr_of_rows])
        for i in range(0, nr_of_rows):
            for j in range(i + 1, nr_of_rows):
                distance_matrix[i, j] = np.sqrt(
                    (
                        (small_dfs[0].iloc[i] - small_dfs[0].iloc[j]) ** 2 +
                        (small_dfs[1].iloc[i] - small_dfs[1].iloc[j]) ** 2 +
                        (small_dfs[2].iloc[i] - small_dfs[2].iloc[j]) ** 2
                    ).sum()
                )
                distance_matrix[j, i] = distance_matrix[i, j]

        result = pd.concat([result, pd.DataFrame(distance_matrix, columns=column_names)])

        # Reappend the categorical columns
        for colname in categorical_colnames:
            result[colname] = dataframes[0][colname]

        return result

    @staticmethod
    def _calculate_distance_with_dtw(data, norm: int = 2):
        """
        Calculates metric for specified dataframe using dynamic time warping utilizing norm.
        :param data:
        :param norm: Defines which L-norm is to be used.
        :return result: A 2D-nd-array containing distance from each segment to each other segment (same as with scipy's
        cdist() - zeros(shape, dtype=float, order='C'))
        """

        # Initialize empty distance matrix.
        dist_matrix = np.zeros((data.shape[0], data.shape[0]), dtype=float)

        # Note regarding multithreading: Splitting up by rows leads to imbalance amongst thread workloads.
        # Instead, we split up all possible pairings to ensure even workloads and collect the results (and assemble
        # the distance matrix) after the threads finished their calculations.
        # Generate all pairings.
        segment_pairings = [(i, j) for i in range(0, data.shape[0]) for j in range(0, data.shape[0]) if j > i]

        # Set up multithreading. Run as many threads as logical cores are available on this machine - 1.
        num_threads = psutil.cpu_count(logical=True)
        threads = []
        for i in range(0, num_threads):
            # Calculate distance with fastDTW between each pairing of segments. Distances between elements to themselves
            # are ignored and hence retain their intial value of 0.
            thread = DTWThread(thread_id=i,
                               num_threads=num_threads,
                               segment_pairings=segment_pairings,
                               distance_matrix=dist_matrix,
                               data_to_process=data,
                               norm=norm)
            threads.append(thread)
            thread.start()

        # Wait for threads to finish.
        for thread in threads:
            thread.join()

        return dist_matrix

    @staticmethod
    def _cut_trip(sensor_data, target_column_name: str, snippet_length=30, column_names=None):
        """
        Helper function to cut one trip into segments of snippet_length
        and return the new pandas.DataFrame that includes the "total"
        of each value.
        :param target_column_name: Name of column to use as observation in trip (i. e. one of: "total", "x", "y", "z").
        """
        HERTZ_RATE = 20
        nr_of_trip_columns = HERTZ_RATE * snippet_length
        categorical_colnames = ["mode","notes","scripted","token", "trip_id"]
        if column_names is None:
            column_names = ["snippet_"+str(i) for i in range(nr_of_trip_columns)]
            column_names = column_names + categorical_colnames

        result = pd.DataFrame(columns=column_names).drop(categorical_colnames, axis=1)
        copied_sensor_data = sensor_data.reset_index(drop=True)
        copied_sensor_data = copied_sensor_data
        end_index = copied_sensor_data.index[-1]
        # // floor division syntax
        # the last segment wich is smaller than 30 seconds will be dropped
        nr_of_rows = end_index // nr_of_trip_columns
        start_index = 0

        for row_index in range(nr_of_rows):
            to_index = start_index + nr_of_trip_columns
            row_i = copied_sensor_data.loc[start_index:to_index-1, target_column_name]
            result.loc[row_index,:] = list(row_i)
            start_index = to_index

        return result

    @staticmethod
    def _get_row_entries_for_trip(trip,  sensor_type="acceleration"):
        """
        Helper function which splits on trip into the four parts
        sensor_data, mode, notes and scripted, where scripted is
        a binary variable where scripted=1 and ordinary=0
        """
        sensor_data, mode, notes, scripted = None, None, None, None
        for table_name, table_content in trip.items():
            if table_name == "sensor":
                sensor_data = table_content[table_content["sensor"] == sensor_type]
            if table_name == "annotation":
                annotation_data = table_content
                mode = annotation_data["mode"][0]
                notes = annotation_data["notes"][0]
                if "scripted" in str(notes).lower():
                    scripted = 1
                else:
                    scripted = 0
        return sensor_data, mode, notes, scripted

    @staticmethod
    def unpack_all_trips(dfs: dict, keep_tokens=False):
        """
        Helper method that takes a dictionary of the trips per token and returns a list
        of all trips. Assumed nested structure is:
        dict[token] = list of trips per token
        :param keep_tokens: bool, default=False,
                    if True, the token is appended to the annotation dataframe.
                    This makes it easier to identify the trips later.
        """
        result = []
        dfs_copy = deepcopy(dfs)
        for token, trips in sorted(dfs_copy.items()):
            if keep_tokens:
                for trip_i in trips:
                    if trip_i["annotation"] is not None:
                        trip_i["annotation"]["token"]=token
            result += trips
        return result


    @staticmethod
    def restore_preprocessed_data_from_disk(filename: str):
        """
        Loads pickled object from disk.
        :param filename: File name/relative path in /data/preprocessed.
        :return: Dictionary holding data for tokens (same format as returned by Preprocessor.preprocess().
        """
        data_dir = DatasetDownloader.get_data_dir()
        full_path = os.path.join(data_dir, "preprocessed", filename)
        with open(full_path, "rb") as file:
            preprocessed_data = file.read()

        # https://media.giphy.com/media/9zXWAIcr6jycE/giphy.gif
        return pickle.loads(preprocessed_data)

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

        trips = {"scripted": {"TRAM": 0, "METRO": 0, "WALK": 0}, "unscripted": {"TRAM": 0, "METRO": 0, "WALK": 0}}
        for i, df in enumerate(dataframes):
            # Assuming "notes" only has one entry per trip and scripted trips' notes contain the word "scripted",
            # while ordinary trips' notes don't.
            if "scripted" in str(df["annotation"]["notes"][0]).lower():
                trips["scripted"][df["annotation"]["mode"][0]] += 1
                for dataframe_name in list_of_dataframe_names_to_cut:
                    # Cut off time series data.
                    dataframes[i][dataframe_name] = Preprocessor._cut_off_start_and_end_in_dataframe(
                        dataframe=df[dataframe_name], cutoff_in_seconds=cutoff_in_seconds
                    )
            else:
                trips["unscripted"][df["annotation"]["mode"][0]] += 1

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
                (dataframe["time"] <= upper_time_threshold - cutoff_in_seconds)
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
        nr_of_recorded_trips = len(all_trips)
        result = pd.DataFrame()
        if convert_time:
            all_trips_copy = Preprocessor.convert_timestamps(all_trips)
        else:
            all_trips_copy = all_trips
        start_times = []
        end_times = []
        for index in range(0, nr_of_recorded_trips):
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
    def _get_shallow_copy(dfs: list):
        """ Helper function to get a shallow copy of the list of dictionaries
            as only sensor data is modified and the rest can be references.
        """
        nr_of_trips = len(dfs)
        result = [{} for trip in range(nr_of_trips)]
        for trip_index, trip_i in enumerate(dfs):
            for key, values in trip_i.items():
                if key == "sensor":
                    result[trip_index][key] = None
                else:
                    result[trip_index][key] = values
        return result


    @staticmethod
    def calculate_paa(dfs, verbose=False):
        newDict = Preprocessor._get_shallow_copy(dfs)
        nr_of_trips = len(dfs)
        for i in range(0, nr_of_trips):
            if verbose:
                print('Frame ', i)
            #get single trip
            sensor_trip = dfs[i]['sensor']
            #get all sensors
            sensor_set = set(sensor_trip['sensor'])
            #create new data frame
            helper = pd.DataFrame()

            for sensor in sensor_set:
                if verbose:
                    print("sensor: ", sensor)

                sensor_data = sensor_trip[sensor_trip['sensor'] == sensor]

                if verbose:
                    print('init time frame')
                    print(Preprocessor.convert_timestamps(sensor_data.head(1)))
                    print(Preprocessor.convert_timestamps(sensor_data.tail(1)))

                sensor_data = sensor_data.drop(['sensor', 'total'], axis=1)
                sensor_data.reset_index(drop=True,inplace=True)
                sensor_data_approximated = Preprocessor.approx_sensor(sensor_data, 100)

                start_index = 0
                stop_index = 1
                end_of_df = len(sensor_data_approximated)

                buffer_helper = pd.DataFrame()
                filler = pd.DataFrame()


                if verbose:
                    print("end_of_df:", end_of_df)

                while stop_index <= end_of_df:
                    if start_index + 30000 <= end_of_df:
                        stop_index = stop_index + 30000
                    else:
                        stop_index = end_of_df+1

                    buffer_helper = Preprocessor.normalize_trip(sensor_data_approximated.iloc[start_index:stop_index,:])

                    filler = filler.append(buffer_helper)
                    start_index = stop_index

                filler['sensor'] = sensor
                filler['total'] = np.linalg.norm(np.array([filler['x'], filler['y'], filler['z']]),ord=2, axis=0)
                helper = pd.concat([helper,filler])

                if verbose:
                    print("complete frame")
                    print(Preprocessor.convert_timestamps(helper.head(1))['time'])
                    print(Preprocessor.convert_timestamps(helper.tail(1))['time'])
                    print('----------------------------')

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

        new_time = np.arange(acc['time'][0], acc['time'][len(acc['time'])-1], atd_ms)
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

        paa = PAA(window_size=5, output_size=None, overlapping=False)
        container = []
        for label in trip.columns:
            # this creates a new object, change to float32 increases speed
            arr = np.array([trip[label]], dtype=np.float64)
            transf = paa.transform(arr)
            container.append(list(transf[0]))

        df = pd.DataFrame(container,trip.columns).T
        df['time'] = [int(i) for i in df['time']]
        return df

    # Note by rmitsch: Commented out since variable 'feature' is not defined. To remove?
    # @staticmethod
    # def plot_paa(sensor_data, w_size=5, seconds=2):
    #
    #
    #     plot_paa(feature, window_size=w_size,output_size=None,overlapping=False,marker='o')

    @staticmethod
    def print_start_and_end_of_recording_per_sensor(df):
        set_of_sensors = set(df['sensor'])
        for sensor in set_of_sensors:
            print("sensor: ", sensor)
            # get all sensor data for specific sensor
            sensor_data = deepcopy(df[df["sensor"] == sensor])

            sensor_data.reset_index(drop=True,inplace=True)

            start = min(sensor_data['time'])
            start = pd.to_datetime(start, unit="ms")
            end = max(sensor_data['time'])
            end =  pd.to_datetime(end, unit="ms")
            print("start of recodring: " , start)
            print("end of recodring: " , end)

    @staticmethod
    def check_second_intervals(sensor_data, seconds=5):
        start = True
        count = 0

        time_series = deepcopy(sensor_data[sensor_data['sensor'] == 'acceleration'])
        time_series.reset_index(drop=True,inplace=True)
        time_series = time_series['time']

        for timestamp in time_series:
            if start:
                print("beginning at:", timestamp.minute,":",timestamp.second)
                prev = timestamp.second
                start=False
            next_time = prev + 1 if prev < 59 else 1

            if timestamp.second == next_time:
                index_pos = time_series[time_series == timestamp].index.tolist()
                print("next second at index: ",index_pos[0], " at time ",timestamp.minute,":",timestamp.second)
                prev = timestamp.second
                count = count + 1

            if count > seconds:
                break
