import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, random_split

from config import paths_manager
from datasources.torchdataset import PowerDataset
from disaggregators.models import SAED
from disaggregators.trainingtools import ClassicTrainingTools
from lab.report import save_report

EPOCHS = 1
WINDOW = 50
BATCH = 1024
appliances = ['microwave1', 'dishwasher1', 'furnace1', 'refrigerator1', 'drye1', 'air1', 'bedroom1', 'bedroom2']

datasets = {}
train_loaders = {}
# for appliance in appliances:
#     dataset = PowerDataset(path='../data/house_7901.csv', device=appliance)
#     train_loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=8)
#     datasets[appliance] = dataset
#     train_loaders[appliance] = train_loader
appliance = 'microwave1'
dataset = PowerDataset(path='../data/house_7901.csv', device=appliance)
valsize = len(dataset) // 5
rem = len(dataset) % 5
train_dataset, val_dataset = random_split(dataset, [4 * valsize, valsize + rem])
train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=BATCH, shuffle=False, num_workers=8)

eval_params = {'device'     : dataset.device,
               'mmax'       : dataset.mmax,
               'groundtruth': ''}
model = SAED(window_size=WINDOW, dropout=0.25)
model_name = model.architecture_name
model = ClassicTrainingTools(model, eval_params)
checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                      dirpath=paths_manager.get_checkpoints_path(appliance, model_name),
                                      filename=f'model-{model.model.architecture_name}' + '-{epoch:02d}-{val_loss:.2f}',
                                      save_top_k=1,
                                      mode='min', )

trainer = pl.Trainer(callbacks=[checkpoint_callback],
                     # default_root_dir='./checkpoints',
                     gpus=1, max_epochs=EPOCHS,
                     auto_lr_find=True)
# trainer.tune(model, train_loader, val_loader)

trainer.fit(model, train_loader, val_loader)
valdata = []
for x, y in val_loader:
    valdata.extend(y.numpy())
valdata = np.array(valdata)
model.set_ground(valdata)
res = trainer.test(model, test_dataloaders=val_loader)
test_result = model.get_res()
results = test_result['metrics']
preds = test_result['preds']
final_experiment_name = 'trained'
save_report(root_dir=paths_manager.get_report_path(appliance, model_name), results=results, preds=preds, ground=valdata)
