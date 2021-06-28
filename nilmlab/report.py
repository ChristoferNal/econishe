import os

import numpy as np
import pandas as pd


def save_report(root_dir=None,
                results={},
                preds=None, ground=None):
    report_filename = 'report.csv'
    data_filename = 'detailed_results.csv'
    print('Report saved at: ', root_dir)

    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    if report_filename in os.listdir(root_dir):
        report = pd.read_csv(os.path.join(root_dir, report_filename))
    else:
        cols = ['recall', 'f1', 'precision',
                'accuracy', 'MAE', 'RETE']
        report = pd.DataFrame(columns=cols)
    report = report.append(results, ignore_index=True)
    report.fillna(np.nan, inplace=True)
    report.to_csv(os.path.join(root_dir, report_filename), index=False)

    cols = ['ground', 'preds']
    res_data = pd.DataFrame(list(zip(ground, preds)),
                            columns=cols)
    res_data.to_csv(os.path.join(root_dir, data_filename), index=False)
