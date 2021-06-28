from nilmmodels.models import SAED, WGRU, Seq2Point
from nilmlab.training_helper import train_val_report
import pytorch_lightning as pl

EPOCHS = 5
WINDOW = 50
BATCH = 1024
appliances = ['microwave1', 'dishwasher1', 'furnace1', 'refrigerator1', 'drye1', 'air1', 'bedroom1', 'bedroom2']
appliances = ['furnace1']
# pl.seed_everything(45)
house_path = '../data/house_7901.csv'
# house_path = '../data/house_5746.csv'
# house_path = '../data/house_8565.csv'

datasets = {}
train_loaders = {}
for appliance in appliances:
    model = SAED(window_size=WINDOW, dropout=0.0)
    # model = Seq2Point(window_size=WINDOW, dropout=0)
    # model = WGRU(dropout=0.25)
    train_val_report(model, house_path, appliance, epochs=EPOCHS, batch=BATCH, window=WINDOW)



#
# datasets = {}
# train_loaders = {}
# for appliance in appliances:
#     # model = SAED(window_size=WINDOW, dropout=0.250)
#     # model = WGRU(dropout=0.25)
#     train_val_report(model, house_path, appliance, epochs=EPOCHS, batch=BATCH, window=WINDOW)
# #
