import os

MODEL_PT = 'model.pt'
dirname = os.path.dirname(__file__)
REPORT = os.path.join(dirname, '../report')
CHECKPOINTS = 'checkpoints'
SAVED_MODELS = os.path.join(dirname, '../saved_models')


def get_report_path(appliance, model_name):
    return os.path.join(REPORT, appliance, model_name)


def get_checkpoints_path(appliance, model_name):
    return os.path.join(get_report_path(appliance, model_name), CHECKPOINTS)


def get_saved_models_path(appliance, model_name):
    path = os.path.join(SAVED_MODELS, appliance, model_name)
    if not os.path.exists(path):
        os.makedirs(path)
    return os.path.join(path, MODEL_PT)