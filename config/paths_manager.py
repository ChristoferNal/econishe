import os

dirname = os.path.dirname(__file__)
REPORT = os.path.join(dirname, '../report')
CHECKPOINTS = 'checkpoints'
SAVED_MODELS = os.path.join(dirname, '../saved_models')


def get_report_path(appliance, model_name):
    return os.path.join(REPORT, appliance, model_name)


def get_checkpoints_path(appliance, model_name):
    return os.path.join(get_report_path(appliance, model_name), CHECKPOINTS)


def get_saved_models_path(appliance, model_name):
    return os.path.join(SAVED_MODELS, appliance, model_name)