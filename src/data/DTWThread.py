import threading
import multiprocessing
import pandas as pd
import numpy as np
from fastdtw import fastdtw


class DTWThread(threading.Thread):
    def __init__(self,
                 thread_id: int,
                 num_threads: int,
                 segment_pairings: list,
                 data_to_process: pd.DataFrame,
                 distance_matrix: np.ndarray,
                 norm: int = 2):
        """
        Initializes thread. Rows to be processed are selected here.
        :param thread_id:
        :param num_threads:
        :param data_to_process:
        :param distance_matrix: 2D-ndarray to store results in.
        :param norm: Integer defining which L-norm to use.
        """
        threading.Thread.__init__(self)

        self.thread_id = thread_id
        self.segment_pairings = segment_pairings
        self.data_to_process = data_to_process
        self.distance_matrix = distance_matrix
        self.norm = norm

        # Calculate number of rows per thread.
        number_of_pairings_per_thread = int(len(segment_pairings) / num_threads)
        # Calculate first row index as first index after all rows processed by previous threads (thread ID equals
        # sequence number).
        first_index = number_of_pairings_per_thread * thread_id
        # Calculate last row index as start index + number of rows per thread if this is not the last thread or last
        # row index in data_to_process if this is the last thread (i. e. last thread calculates more lines if there's
        # an uneven number of records).
        last_index = first_index + number_of_pairings_per_thread - 1 \
            if thread_id < num_threads - 1 \
            else len(segment_pairings) - 1

        # Store first and last line to process.
        self.interval = (first_index, last_index)

    def run(self):
        """
        Run DTW on part of the specified dataframe.
        :return:
        """
        # Calculate distance with fastDTW between each pairing of segments. Distances between elements to themselves
        # are ignored and hence retain their intial value of 0.
        for i, segment_pairing in enumerate(self.segment_pairings[self.interval[0]:(self.interval[1] + 1)]):
            # distance, cost, acc, path = dtw(
            #     self.data_to_process.iloc[segment_pairing[0]].values.reshape(-1, 1),
            #     self.data_to_process.iloc[segment_pairing[1]].values.reshape(-1, 1),
            #     dist=lambda x, y: np.linalg.norm(x - y, ord=self.norm)
            # )

            distance, path = fastdtw(
                self.data_to_process.iloc[segment_pairing[0]].values.reshape(-1, 1),
                self.data_to_process.iloc[segment_pairing[1]].values.reshape(-1, 1),
                dist=self.norm
            )

            self.distance_matrix[segment_pairing[0], segment_pairing[1]] = distance
            self.distance_matrix[segment_pairing[1], segment_pairing[0]] = distance
