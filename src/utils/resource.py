"""

* Mlflow Wrapper Class
* Class provides popular methods in mlflow

"""

from mlflow.tracking import MlflowClient
from omegaconf import DictConfig, ListConfig


class MlflowWriter():
    def __init__(self, experiment_name):
        self.client = MlflowClient()
        try:
            self.experiment_id = self.client.create_experiment(experiment_name)
        except Exception as e:
            print(e)
            self.experiment_id = self.client.get_experiment_by_name(experiment_name).experiment_id

        self.experiment = self.client.get_experiment(self.experiment_id)
        print("Name: {}".format(self.experiment.name))
        print("Experiment_id: {}".format(self.experiment.experiment_id))
        print("Artifact Loc: {}".format(self.experiment.artifact_location))

    def log_params_from_omegaconf_dict(self, params):
        for param_name, element in params.items():
            self._explore_recursive(param_name, element)

    def _explore_recursive(self, parent_name, element):
        if isinstance(element, DictConfig):
            for k, v in element.items():
                if isinstance(v, DictConfig) or isinstance(v, ListConfig):
                    self._explore_recursive(f'{parent_name}.{k}', v)
                else:
                    self.client.log_param(self.run_id, f'{parent_name}.{k}', v)
        elif isinstance(element, ListConfig):
            for i, v in enumerate(element):
                self.client.log_param(self.run_id, f'{parent_name}.{i}', v)
        else:
            self.client.log_param(self.run_id, f'{parent_name}', element)

    def log_param(self, key, value):
        self.client.log_param(self.run_id, key, value)

    def log_metric(self, key, value):
        self.client.log_metric(self.run_id, key, value)

    def log_metric_step(self, key, value, step):
        self.client.log_metric(self.run_id, key, value, step=step)

    def log_artifact(self, local_path):
        self.client.log_artifact(self.run_id, local_path)

    def log_dict(self, dictionary, file):
        self.client.log_dict(self.run_id, dictionary, file)

    def log_figure(self, figure, file):
        self.client.log_figure(self.run_id, figure, file)

    def set_terminated(self):
        self.client.set_terminated(self.run_id)

    def create_new_run(self, tags=None):
        self.run = self.client.create_run(self.experiment_id, tags=tags)
        self.run_id = self.run.info.run_id
