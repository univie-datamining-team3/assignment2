from sklearn import metrics
import pandas as pd
import numpy as np
from copy import deepcopy
import os
import re
from io import StringIO

from subprocess import Popen, PIPE
from data.download import DatasetDownloader
from data.preprocessing import Preprocessor
import time
from sklearn.cluster import KMeans
import plotly.offline as plotly_offline
import plotly.graph_objs as go
import matplotlib.pyplot as plt


class ElkiPipe:
    """
    This class provides a pipe to the java implementation of elki:
    https://elki-project.github.io/
    """
    def __init__(self, elki_path_to_jar=None):
        if elki_path_to_jar is None:
            self.elki_path_to_jar = os.path.join("models","elki-bundle-0.7.1.jar")

    def run_elki(self, df:pd.DataFrame, parameters:list, plot_path=None):
        """
        Wrapper function to elki java implementation.
        Based on https://stackoverflow.com/questions/15326505/running-clustering-algorithms-in-elki and
        the command line builder from https://elki-project.github.io/
        e.g. For predecon we have the following command:
        KDDCLIApplication -dbc.in filepath -algorithm clustering.subspace.PreDeCon -dbscan.epsilon 10.0 -dbscan.minpts 5 \
         -predecon.delta 0.1 -predecon.lambda 2

        """

        if plot_path is None:
            elki_data = self.__remove_unused_columns(df)
        else:
            elki_data = df

        temp_path = "elki_temp.csv"
        elki_data.to_csv(temp_path, sep=";", index=False)

        constant_args = ["java", "-jar", self.elki_path_to_jar,
                         "KDDCLIApplication",
                         "-dbc.in", temp_path,
                         "-parser.colsep", ";"]

        args = constant_args + parameters
        if plot_path is not None:
            args += ["-resulthandler", "ExportVisualizations",
                     "-vis.output", plot_path]

        process = Popen(args,
                        stdout=PIPE)
        (console_output, error) = process.communicate()
        exit_code = process.wait()
        if exit_code != 0:
            raise IOError ("ELKI failed to run:\n{}".format(console_output.decode("utf-8", errors='ignore')))

        #clean up file
        os.remove(temp_path)

        clustering_results = console_output.decode("utf-8")

        #process console output
        cleaned_results = self.__get_elki_cluster_result(clustering_results)
        return cleaned_results

    @staticmethod
    def __get_elki_cluster_result(results):
        """ Constructs a dataframe with the cluster labels
        """
        print(results)
        cleaned_results = pd.DataFrame(columns=["ID","label"])
        for cluster_id, points_in_cluster in enumerate(results.split("Cluster: Cluster")[1:]):
            helper_df = pd.DataFrame(columns=["ID","label"])

            # Important for sorting the points to its original order
            point_ids = re.findall(r"ID=(\d+)", points_in_cluster)
            helper_df["ID"] = point_ids
            helper_df["label"] = cluster_id
            cleaned_results = pd.concat([cleaned_results, helper_df])

        # Prepare labels
        cleaned_results = cleaned_results.sort_values("ID").drop("ID",axis=1).reset_index(drop=True)
        return cleaned_results

    @staticmethod
    def get_parameters_for_predecon(param_eps = 10.0, param_minpts = 2, param_delta = 0.1, param_lambda = 1, param_kappa = 20.0):
        """
        This method returns the parameters for the PreDeCon algorithm in format for the Elki Pipe
            # Hyperparameter param_minpts: Minimal number of points in epsilon-neighbourhood.
            # Hyperparameter param_eps: Distance for neighbourhood.
            # Hyperparameter param_delta : Variance threshold for subspace preference clusters.
            # Hyperparameter param_lambda: Dimensionality threshold.
            # Hyperparameter param_kappa: Weight for subspace preference vectors.
        """

        return [ "-algorithm", "clustering.subspace.PreDeCon",
                 "-dbscan.epsilon", str(float(param_eps)),
                 "-dbscan.minpts", str(param_minpts),
                 "-predecon.delta", str(float(param_delta)),
                 "-predecon.lambda", str(param_lambda),
                 "-predecon.kappa", str(float(param_kappa))
                ]

    @staticmethod
    def get_parameters_for_hdbscan(param_minpts=100):
        return ["-algorithm", "clustering.hierarchical.HDBSCANLinearMemory",
                "-hdbscan.minPts", str(param_minpts)]

    @staticmethod
    def __remove_unused_columns(df:pd.DataFrame, remove_cols = None):
        if remove_cols is None:
            remove_cols = ["mode", "notes", "scripted", "token", "trip_id"]
        colnames = list(df.columns)
        df_copy = deepcopy(df)
        for col_i in remove_cols:
            if col_i in colnames:
                df_copy = df_copy.drop(col_i,axis=1)

        return df_copy

class Clustering:
    """
    Class for routines related to clustering.
    """

    # All distance metrics for numerical data natively supported by scipy.
    # See https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html.
    SCIPY_DISTANCE_METRICS = [
        "braycurtis", "canberra", "chebyshev", "cityblock", "correlation", "cosine",
        "euclidean", "hamming", "mahalanobis", "matching", "minkowski", "seuclidean",
        "sqeuclidean"
    ]

    @staticmethod
    def evaluate_distance_metrics(filename: str = "preprocessed_data.dat"):
        """
        Evaluates the following distance metrics...
        ...for column "total":
            - All of scipy's built-in distance metrics
              (https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html),
            - DTW.
        ...for individual columns (as described in task assignment):
            - Euclidean distance.
        See Preprocessor.calculate_distance_for_individual_columns() on why only L2 is currently supported for indiv.
        columns.
        :param filename:
        :return:
        """

        # Auxiliary variable holding names of columns with categorical data.
        categorical_columns = ["mode", "notes", "scripted", "token", "trip_id"]

        # 1. Load preprocessed data from disk.
        data_dir = os.path.join(os.path.abspath(DatasetDownloader.get_data_dir()))
        file_path = os.path.join(data_dir, "preprocessed", filename)
        dfs = Preprocessor.restore_preprocessed_data_from_disk(file_path)

        # 2. Get cut trip snippets.
        trips_cut_per_30_sec = Preprocessor.get_cut_trip_snippets_for_targets(
            dfs=dfs,
            snippet_length=30,
            sensor_type="acceleration",
            target_column_names=["total", "x", "y", "z"]
        )

        ##############################################################################
        # 3. Calculate distance matrices for all metrics with column "total".
        ##############################################################################

        all_dist_metrics_for_total = deepcopy(Clustering.SCIPY_DISTANCE_METRICS)
        all_dist_metrics_for_total.append("dtw")
        performance_data = []

        for index, metric in enumerate(all_dist_metrics_for_total):
            if metric not in ("metric_name_to_exclude",):
                print("Calculating ", metric)
                start_time = time.time()
                distance_matrix_with_categorical_data = Preprocessor.calculate_distance_for_n2(
                    trips_cut_per_30_sec[0],
                    metric=metric
                )
                runtime_in_sec = (time.time() - start_time) / 1000

                # Drop categorical data.
                distance_matrix = distance_matrix_with_categorical_data.drop(categorical_columns, axis=1)

                # Naive asumption of 3 clusters due to 3 labels Walk, Metro and Tram
                kmeans = KMeans(n_clusters=3, random_state=0).fit(distance_matrix)
                cluster_labels = kmeans.labels_
                distance_matrix_with_categorical_data["cluster_labels"] = cluster_labels

                # Validate performance against assume ground truth of transport modes (e. g. three clusters), ignoring
                # tokens, scripted/unscripted and other potential subclasses.
                performance = Clustering.get_clustering_performance_as_dict(
                    features=distance_matrix,
                    cluster_assignments=cluster_labels,
                    true_labels=distance_matrix_with_categorical_data["mode"]
                )
                performance["runtime_for_distance_calculation"] = float(runtime_in_sec)
                performance["distance_metric"] = metric
                performance["distance_metric_index"] = index + 1
                performance_data.append(performance)

        # Transform performance data to data frame.
        performance_df = pd.DataFrame(performance_data)

        # Plot data as parallel coordinates plot with plotly.
        plotly_data = [
            go.Parcoords(
                line=dict(
                    color=performance_df["distance_metric_index"],
                    #colorscale='Jet',
                    showscale=True
                ),
                dimensions=list([
                    dict(range=[0, 1],
                         constraintrange=[0, 1],
                         label='Adjusted MI', values=performance_df['adjusted_mi']),
                    dict(range=[0, 1],
                         constraintrange=[0, 1],
                         label='Completeness', values=performance_df['completeness']),
                    dict(range=[0, 1],
                         constraintrange=[0, 1],
                         label='Homogeneity', values=performance_df['homogeneity']),
                    dict(range=[0, 1],
                         constraintrange=[0, 1],
                         label='Silhouette score', values=performance_df['silhouette_score']),
                    dict(range=[0, 1],
                         constraintrange=[0, 1],
                         label='V-measure', values=performance_df['v_measure']),
                    dict(range=[0, 0.1],
                         constraintrange=[0, 0.1],
                         label='Runtime', values=performance_df['runtime_for_distance_calculation'])
                ])
            )
        ]

        layout = go.Layout(
            plot_bgcolor='#E5E5E5',
            paper_bgcolor='#E5E5E5'
        )

        fig = go.Figure(data=plotly_data, layout=layout)
        plotly_offline.plot(fig, filename='parcoords-basic.html')

        # Plot data as small multiples with barcharts.
        performance_df.set_index("distance_metric", inplace=True)

        performance_df.drop(
            ["distance_metric_index",
             "true_number_of_clusters",
             "estimated_number_of_clusters",
             "runtime_for_distance_calculation"
             ],
            axis=1,
            inplace=True
        )
        performance_df.plot.barh(stacked=False, grid=True, use_index=True)
        plt.show()

    @staticmethod
    def calculate_silhouette_score(features, cluster_assignments):
        """
        Returns string with Silhouette Coefficient
        """
        if len(set(cluster_assignments)) > 1:
            result = ("Silhouette Coefficient: %0.3f"
                      % metrics.silhouette_score(features, cluster_assignments))
        else:
            result = "Silhouette Coefficient: cannot be calculated"

        return result

    @staticmethod
    def get_clustering_performance(features, cluster_assignments, true_labels=None):
        """
        Calculate several performance measures for clustering
        This code has been adapted from:
        http://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html
        """

        summary = []
        if true_labels is not None:
            # Number of clusters in true_labels, ignoring noise if present.
            n_clusters_ = len(set(true_labels)) - (1 if -1 in true_labels else 0)
            summary.append(('Estimated number of clusters: %d' % len(set(list(cluster_assignments)))))
            summary.append(('True number of clusters: %d' % n_clusters_))
            summary.append(("Homogeneity: %0.3f" % metrics.homogeneity_score(cluster_assignments, true_labels)))
            summary.append(("Completeness: %0.3f" % metrics.completeness_score(cluster_assignments, true_labels)))
            summary.append(("V-measure: %0.3f" % metrics.v_measure_score(cluster_assignments, true_labels)))
            summary.append(
                ("Adjusted MI: %0.3f" % metrics.adjusted_mutual_info_score(true_labels, cluster_assignments)))
            summary.append(calculate_silhouette_score(features, cluster_assignments))
        else:
            summary.append(calculate_silhouette_score(features, cluster_assignments))

        return summary

    @staticmethod
    def get_clustering_performance_as_dict(features, cluster_assignments, true_labels=None):
        """
        Calculate several performance measures for clustering
        This code has been adapted from:
        http://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html
        """

        summary = {}

        if true_labels is not None:
            # Number of clusters in true_labels, ignoring noise if present.
            n_clusters_ = len(set(true_labels)) - (1 if -1 in true_labels else 0)
            summary["estimated_number_of_clusters"] = len(set(list(cluster_assignments)))
            summary["true_number_of_clusters"] = n_clusters_
            summary["homogeneity"] = metrics.homogeneity_score(cluster_assignments, true_labels)
            summary["completeness"] = metrics.completeness_score(cluster_assignments, true_labels)
            summary["v_measure"] = metrics.v_measure_score(cluster_assignments, true_labels)
            summary["adjusted_mi"] = metrics.adjusted_mutual_info_score(true_labels, cluster_assignments)
            summary["silhouette_score"] = calculate_silhouette_score(features, cluster_assignments, as_numeric=True)
        else:
            summary["silhouette_score"] = calculate_silhouette_score(features, cluster_assignments, as_numeric=True)

        return summary

"""
Functions below here: Kept in order not to break backwards compatibility.
"""

def calculate_silhouette_score(features, cluster_assignments, as_numeric=False):
    """
    Returns string with Silhouette Coefficient
    """
    if len(set(cluster_assignments)) > 1:
        if not as_numeric:
            result = ("Silhouette Coefficient: %0.3f"
                  % metrics.silhouette_score(features, cluster_assignments))
        else:
            result = metrics.silhouette_score(features, cluster_assignments)
    else:
        result = ("Silhouette Coefficient: cannot be calculated")

    return result

def get_clustering_performance(features, cluster_assignments, true_labels=None):
    """
    Calculate several performance measures for clustering
    This code has been adapted from:
    http://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html
    """

    summary = []
    if true_labels is not None:
        # Number of clusters in true_labels, ignoring noise if present.
        n_clusters_ = len(set(true_labels)) - (1 if -1 in true_labels else 0)
        summary.append(('Estimated number of clusters: %d' % len(set(list(cluster_assignments)))))
        summary.append(('True number of clusters: %d' % n_clusters_))
        summary.append(("Homogeneity: %0.3f" % metrics.homogeneity_score(cluster_assignments, true_labels)))
        summary.append(("Completeness: %0.3f" % metrics.completeness_score(cluster_assignments, true_labels)))
        summary.append(("V-measure: %0.3f" % metrics.v_measure_score(cluster_assignments, true_labels)))
        summary.append(("Adjusted MI: %0.3f" % metrics.adjusted_mutual_info_score(true_labels,cluster_assignments)))
        summary.append(calculate_silhouette_score(features, cluster_assignments))
    else:
        summary.append(calculate_silhouette_score(features, cluster_assignments))

    return summary
