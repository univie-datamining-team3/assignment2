from bayes_opt import BayesianOptimization
from .TSNEModel import TSNEModel
import pandas
import sobol_seq
import copy
from src.data.preprocessing import DatasetDownloader
import os
import pickle


class BayesianTSNEOptimizer:
    """
    Wrapper class for Bayesian optmization of t-SNE models.
    """

    def __init__(self, high_dim_data, parameters: dict):
        """
        Initializes BO for t-SNE.
        :param high_dim_data:
        :param parameters: Set of parameters. If value for key is scalar, then the parameter is considered fixed. If
        it's a 2-tupel, the parameter will be varied inside the defined ranges.
        """
        self.high_dim_data = high_dim_data

        # Define dictionary holding fixed parameters.
        self.fixed_parameters = {}
        # Define dictionary holding variable parameters.
        self.variable_parameters = {}
        # Store parameters values.
        self.parameters = parameters

        # Store latest t-SNE results.
        self.tsne_results = None

    def run(self, num_iterations: int, init_values: dict = None, kappa: int = 10):
        """
        Fetches latest t-SNE model from DB. Collects pickled BO status object, if existent.
        Intermediate t-SNE models are persisted.
        :param num_iterations: Number of iterations BO should run.
        :param init_values: Ditionary with initialization values (strucutured like BayesianOptimization.res['all']).
        :param kappa:
        :return: BayesianOptimization.res['all'].
        """

        # Process set parameters.
        for param, param_extrema in self.parameters.items():
            if len(param_extrema) > 1:
                self.variable_parameters[param] = param_extrema
            else:
                self.fixed_parameters[param] = param_extrema[0]

        # Create optimization object.
        bo = BayesianOptimization(self._calculate_tsne_quality, self.variable_parameters)

        # Pass previous results to BO instance.
        if init_values is not None:
            bo.initialize(self._cast_results_to_initialization_format(init_values=init_values))

        # 4. Execute optimization.
        num_init_points = max(int(num_iterations / 4), 1)
        bo.maximize(init_points=num_init_points, n_iter=(num_iterations - num_init_points), kappa=kappa, acq='ucb')

        # 5. Print results.
        return bo.res["all"]

    def _cast_results_to_initialization_format(self, init_values: dict):
        """
        Transforms dataset structured like BayesianOptimization.res['all']) to dataset accepted by
        BayesianOptimization.initialize().
        :param init_values:
        :return:
        """

        # Set up dictionary holding values.
        cast_init_values = {}
        keys = set()
        for key in self.variable_parameters:
            keys.add(key)
        keys.add("target")

        # Append values to new structure.
        for key in keys:
            # Add new list, if key doesn't exist yet.
            if key not in cast_init_values:
                cast_init_values[key] = []
            # Add new values for this parameter.
            for i in range(0, len(init_values["values"])):
                if key is not "target":
                    cast_init_values[key].append(init_values["params"][i][key])
                else:
                    cast_init_values[key].append(init_values["values"][i])

        return cast_init_values

    def _calculate_tsne_quality(self, **kwargs):
        """
        Optimization target. Wrapper for generating t-SNE models.
        :param kwargs: Dictionary holding some of the following data points for this run in general and its historical
        t-SNE results (tm.X denotes a t-SNE model's property, r.X a run's property):
            tm.n_components,
            tm.perplexity,
            tm.early_exaggeration,
            tm.learning_rate,
            tm.n_iter,
            tm.min_grad_norm,
            tm.metric,
            tm.init_method,
            tm.random_state,
            tm.angle,
            tm.measure_trustworthiness,
            tm.measure_continuYity,
            tm.measure_generalization_accuracy,
            tm.measure_word_embedding_information_ratio,
            tm.measure_user_quality
        A union of the dynamic parameters specified by BO and the static, user-defined ones has to yield a dictionary
        containing all of the above parameters.
        :return:
        """
        # Fetch parameters defined by BO.
        parameters = {}
        parameters.update(kwargs)

        ################################
        # 1. Prepare parameter set.
        ################################

        # Add missing (fixed) parameters; rename to avoid name conflicts (unify names).
        parameters = {**parameters, **self.fixed_parameters}

        # If categorical attributes are to be varied: Translate floating point value specified by BO to category string
        # accepted by sklearn's t-SNE.
        for key in parameters:
            if key in ["metric", "init_method"]:
                parameters[key] = TSNEModel.CATEGORICAL_VALUES[key][int(round(parameters[key]))]
            # Cast integer values back to integer.
            if key in ["n_iter", "random_state", "n_components"]:
                parameters[key] = int(parameters[key])

        ################################
        # 2. Calculate t-SNE model.
        ################################

        tsne_model = TSNEModel(
            num_dimensions=parameters["n_components"],
            perplexity=parameters["perplexity"],
            early_exaggeration=parameters["early_exaggeration"],
            learning_rate=parameters["learning_rate"],
            num_iterations=parameters["n_iter"],
            min_grad_norm=parameters["min_grad_norm"],
            random_state=parameters["random_state"],
            angle=parameters["angle"],
            metric=parameters["metric"],
            init_method=parameters["init_method"]
        )
        self.tsne_results = tsne_model.run(self.high_dim_data)

        ################################
        # 3. Calculate t-SNE model's quality and update database records.
        ################################

        quality_measures = tsne_model.calculate_quality_measures(high_dim_data=self.high_dim_data)

        ################################
        # 5. Return model score.
        ################################

        return (
            quality_measures["trustworthiness"] +
            quality_measures["continuity"] +
            quality_measures["lcmc"]
        ) / 3.0

    @staticmethod
    def generate_initial_parameter_sets(fixed_parameters, num_iter=10):
        """
        Generates initial t-SNE parameter samples based on Sobol sequences.
        :param fixed_parameters:
        :param num_iter:
        :return: List of dictionaries with valid t-SNE parameter sets.
        """
        parameter_sets = []

        # Calculate Sobol sequence numbers.
        sobol_numbers = sobol_seq.i4_sobol_generate(10 - len(fixed_parameters), num_iter)

        #  Create parameter range dictionary w/o entries for fixed parameters.
        param_ranges = copy.deepcopy(TSNEModel.PARAMETER_RANGES)
        for key in fixed_parameters.keys():
            param_ranges.pop(key, None)

        for curr_iter in range(0, num_iter):
            parameter_set = {}

            for param_entry, sobol_number in zip(param_ranges.items(), sobol_numbers[curr_iter]):
                param = param_entry[0]
                param_range = param_entry[1]

                # Categorical values.
                if param in ("init_method", "metric") and param not in fixed_parameters:
                    index = param_range[0] + round((param_range[1] - param_range[0]) * sobol_number)
                    parameter_set[param] = TSNEModel.CATEGORICAL_VALUES[param][int(index)]

                # Numerical values.
                elif param not in fixed_parameters:
                    # Integer values.
                    if param in ["n_components", "random_state", "n_iter"]:
                        parameter_set[param] = int(
                            param_range[0] + round((param_range[1] - param_range[0]) * sobol_number)
                        )
                    # Float values.
                    else:
                        parameter_set[param] = param_range[0] + (param_range[1] - param_range[0]) * sobol_number

            # Append merged parameter set including fixed parameters.
            parameter_sets.append({**parameter_set, **fixed_parameters})

        return parameter_sets

    @staticmethod
    def persist_result_dict(results: dict, filename: str):
        """
        Stores pickled result dictionary in /data/models.
        :param results:
        :param filename:
        :return:
        """
        data_dir = DatasetDownloader.get_data_dir()
        folder_path = os.path.join(data_dir, "models")

        # make sure the directory exists
        DatasetDownloader.setup_directory(folder_path)
        full_path = os.path.join(folder_path, filename)

        # Dump file.
        with open(full_path, "wb") as file:
            file.write(pickle.dumps(results))

    @staticmethod
    def load_result_dict(filename: str):
        """
        Loads pickled result dictionary from /data/models.
        :param filename:
        :return: Unpickled dictionary. None if file doesn't exist.
        """
        data_dir = DatasetDownloader.get_data_dir()
        full_path = os.path.join(data_dir, "models", filename)

        try:
            with open(full_path, "rb") as file:
                tsne_result_data = file.read()

            return pickle.loads(tsne_result_data)

        except FileNotFoundError:
            return None

    @staticmethod
    def merge_result_dictionaries(result_dict1: dict, result_dict2: dict):
        """
        Merges two result dictionaries generated by module bayes_opt.
        :param result_dict1:
        :param result_dict2:
        :return: Merged dictionary.
        """

        merged_result = copy.deepcopy(result_dict1)

        # Order is irrelevant.
        merged_result["values"].extend(result_dict2["values"])
        merged_result["params"].extend(result_dict2["params"])

        return merged_result
