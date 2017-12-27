from models.dimensionality_reduction.BayesianTSNEOptimizer import BayesianTSNEOptimizer
from models.dimensionality_reduction.TSNEModel import TSNEModel
import os
from src.data.download import DatasetDownloader
from src.data.preprocessing import Preprocessor
import copy


param_ranges = copy.deepcopy(TSNEModel.PARAMETER_RANGES)
# Fix some parameters.
param_ranges["metric"] = (TSNEModel.CATEGORICAL_VALUES["metric"].index("precomputed"),)
param_ranges["init_method"] = (TSNEModel.CATEGORICAL_VALUES["init_method"].index("random"),)
param_ranges["random_state"] = (42,)


# Load data from disk.
data_dir = os.path.join(os.path.abspath(DatasetDownloader.get_data_dir()))
file_path = os.path.join(data_dir, "preprocessed","preprocessed_data.dat")
dfs = Preprocessor.restore_preprocessed_data_from_disk(file_path)

# Calculate distances.
trips_cut_per_30_sec = Preprocessor.get_cut_trip_snippets_for_total(dfs)
euclidean_distances = Preprocessor.calculate_distance_for_n2(trips_cut_per_30_sec, metric="euclidean")

# Remove cat. columns.
categorical_columns = ["mode", "notes", "scripted", "token", "trip_id"]
segment_distance_matrix = euclidean_distances.drop(categorical_columns,axis=1)

# Initialize new BO object.
boOpt = BayesianTSNEOptimizer(high_dim_data=segment_distance_matrix, parameters=param_ranges)

# Load existing results.
history = BayesianTSNEOptimizer.load_result_dict("tsne_results")
# Execute optimization; initialize with existing results.
results = boOpt.run(num_iterations=20, init_values=history)
# Save merged result set (new results and existing ones).
BayesianTSNEOptimizer.persist_result_dict(
    results=BayesianTSNEOptimizer.merge_result_dictionaries(results, history),
    filename="tsne_results"
)
