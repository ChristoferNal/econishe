from disaggregators.models import SAED, WGRU
from lab.training_helper import train_val_report
import pytorch_lightning as pl

EPOCHS = 10
WINDOW = 50
BATCH = 1024
appliances = ['microwave1', 'dishwasher1', 'furnace1', 'refrigerator1', 'drye1', 'air1', 'bedroom1', 'bedroom2']
pl.seed_everything(42)
house_path = '../data/house_7901.csv'

datasets = {}
train_loaders = {}
for appliance in appliances:
    model = SAED(window_size=WINDOW, dropout=0.250)
    # model = WGRU(dropout=0.25)
    train_val_report(model, house_path, appliance, epochs=EPOCHS, batch=BATCH, window=WINDOW)

# appliance = 'microwave1'
# model = SAED(window_size=WINDOW, dropout=0.250)
# # model = WGRU(dropout=0.25)
# train_val_report(model, house_path, appliance, epochs=EPOCHS, batch=BATCH, window=WINDOW)
