from sklearn.manifold import TSNE
import numpy
import coranking
from coranking.metrics import trustworthiness, continuity, LCMC
import hdbscan
import scipy


class TSNEModel:
    """
    Representation of t-SNE model including configuration and actual data.
    """

    # Define possible values for categorical variables.
    CATEGORICAL_VALUES = {
        'metric': ["braycurtis", "canberra", "chebyshev", "cityblock", "correlation", "cosine",
                   "dice", "euclidean", "hamming", "jaccard", "kulsinski", "mahalanobis",
                   "matching", "minkowski", "rogerstanimoto", "russellrao", "seuclidean",
                   "sokalmichener", "sokalsneath", "sqeuclidean", "yule", "precomputed"],
        'init_method': ["random", "pca"]
    }

    # Hardcode thresholds for parameter values. Categorical values are represented by indices.
    PARAMETER_RANGES = {
        "n_components": (1, 3),
        "perplexity": (5, 50),
        "early_exaggeration": (1, 50),
        "learning_rate": (10, 1000),
        "n_iter": (250, 10000),
        "min_grad_norm": (0.0000001, 0.1),
        "metric": (0, 21),
        "init_method": (0, 1),
        "random_state": (1, 100),
        "angle": (0.1, 0.9)
    }

    def __init__(self,
                 num_dimensions,
                 perplexity,
                 early_exaggeration,
                 learning_rate,
                 num_iterations,
                 min_grad_norm,
                 random_state,
                 angle,
                 metric,
                 init_method):
        """
        :param num_dimensions:
        :param perplexity:
        :param early_exaggeration:
        :param learning_rate:
        :param num_iterations:
        :param min_grad_norm:
        :param random_state:
        :param angle:
        :param metric:
        :param init_method:
        """

        self.num_dimensions = num_dimensions
        self.perplexity = perplexity
        self.early_exaggeration = early_exaggeration
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.min_grad_norm = min_grad_norm
        self.random_state = random_state
        self.angle = angle
        self.metric = metric
        self.init_method = init_method

        self.tsne_results = None
        self.high_dim_data = None

    def run(self, high_dim_data):
        """
        Runs t-SNE model with specified parameters and data. Returns result.
        :param high_dim_data: High-dim. data.
        :return:
        """
        self.high_dim_data = high_dim_data

        # Initialize t-SNE instance.
        tsne = TSNE(
            n_components=self.num_dimensions,
            perplexity=self.perplexity,
            early_exaggeration=self.early_exaggeration,
            learning_rate=self.learning_rate,
            n_iter=self.num_iterations,
            min_grad_norm=self.min_grad_norm,
            random_state=self.random_state,
            angle=self.angle,
            metric=self.metric,
            init=self.init_method)

        # Train TSNE model.
        self.tsne_results = tsne.fit_transform(high_dim_data)

        return self.tsne_results

    def calculate_quality_measures(self, high_dim_data, cluster_memberships: numpy.ndarray):
        """
        Calculates quality measures for specified t-SNE result data.
        :param high_dim_data:
        :param cluster_memberships: List of cluster memberships for each datapoint. Values of elements are irrelevant,
        only (in-)equality of values is considered.
        :return:
        """

        # 1. Calculate coranking matrix.
        coranking_matrix = coranking.coranking_matrix(high_dim_data, self.tsne_results).astype(numpy.float16)

        # 2. Calculate trustworthiness.
        trust = trustworthiness(coranking_matrix, min_k=99, max_k=100)

        # 3. Calculate continuity.
        cont = continuity(coranking_matrix, min_k=99, max_k=100)

        # 4. Calculate LCMC.
        lcmc = LCMC(coranking_matrix, min_k=99, max_k=100)

        # 5. Calculate cluster entropy based on specified ground-truth cluster memberships.

        # Fit HDBSCAN cluster model.
        clusterer = hdbscan.HDBSCAN(min_cluster_size=50, gen_min_span_tree=True)
        clusterer.fit(high_dim_data)

        # Calculate entropy.
        entropy = 0
        for i in range(0, max(clusterer.labels_) + 1):
            # Get metadata for all segments in this cluster.
            cluster_members = cluster_memberships[clusterer.labels_ == i]

            # Calculate intra-cluster entropy based on cluster membership (e. g. transport mode).
            values, gt_cluster_counts_in_model_cluster = numpy.unique(cluster_members, return_counts=True)

            # Calculate total (weighted by relative cluster size) entropy.
            entropy += scipy.stats.entropy(gt_cluster_counts_in_model_cluster, base=2) * \
                       numpy.sum(gt_cluster_counts_in_model_cluster) / high_dim_data.shape[0]

        return {
            "trustworthiness": float(trust[0]),
            "continuity": float(cont[0]),
            "lcmc": float(lcmc[0]),
            "entropy": float(entropy)
        }
