from sklearn import metrics


def calculate_silhouette_score(features, cluster_assignments):
    """
    Returns string with Silhouette Coefficient
    """
    if len(set(cluster_assignments)) > 1:
        result = ("Silhouette Coefficient: %0.3f"
              % metrics.silhouette_score(features, cluster_assignments))
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
        summary.append(('True number of clusters: %d' % len(set(list(cluster_assignments)))))
        summary.append(('Estimated number of clusters: %d' % n_clusters_))
        summary.append(("Homogeneity: %0.3f" % metrics.homogeneity_score(cluster_assignments, true_labels)))
        summary.append(("Completeness: %0.3f" % metrics.completeness_score(cluster_assignments, true_labels)))
        summary.append(("V-measure: %0.3f" % metrics.v_measure_score(cluster_assignments, true_labels)))
        summary.append(calculate_silhouette_score(features, cluster_assignments))
    else:
        summary.append(calculate_silhouette_score(features, cluster_assignments))


    return summary
