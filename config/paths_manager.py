import os

dirname = os.path.dirname(__file__)
REPORT = os.path.join(dirname, '../report')
CHECKPOINTS = 'checkpoints'


def get_report_path(appliance, model_name):
    return os.path.join(REPORT, appliance, model_name)


def get_checkpoints_path(appliance, model_name):
    return os.path.join(get_report_path(appliance, model_name), CHECKPOINTS)
