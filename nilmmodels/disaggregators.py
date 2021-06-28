from abc import abstractmethod, ABC

import torch

from config import paths_manager
from nilmmodels.models import SAED


class Dissagregator(ABC):

    def __init__(self, appliance, model_name, window):
        self.model = self.build_and_load(appliance, model_name, window)

    def disaggregate(self, data):
        return self.model.eval(data)

    @staticmethod
    def load_model(appliance, model, model_name):
        return model.load_state_dict(torch.load(paths_manager.get_saved_models_path(appliance, model_name)))

    @abstractmethod
    def build_and_load(self, appliance, model_name, window):
        pass


class SAEDDissagregator(Dissagregator):

    def __init__(self, appliance, model_name, window):
        super().__init__(appliance, model_name, window)

    def build_and_load(self, appliance, model_name, window):
        model = SAED(window_size=window, dropout=0.0)
        return self.load_model(appliance, model, model_name)
