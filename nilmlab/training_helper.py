import os

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split

from config import paths_manager
from datasources.torchdataset import PowerDataset
from nilmlab.trainingtools import ClassicTrainingTools
from nilmlab.report import save_report


def train_val_report(model, house_path, appliance, epochs, batch, window, val=False):
    dataset = PowerDataset(path=house_path, device=appliance, window_size=window)
    val_loader = None
    if val:
        valsize = len(dataset) // 5
        rem = len(dataset) % 5
        train_dataset, val_dataset = random_split(dataset, [4 * valsize, valsize + rem])
        train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True, num_workers=8)
        val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=False, num_workers=8)
    else:
        train_loader = DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=8)

    eval_params = {'device'     : dataset.device,
                   'mmax'       : dataset.mmax,
                   'groundtruth': ''}
    model_name = model.architecture_name
    model = ClassicTrainingTools(model, eval_params)
    # checkpoint_callback = ModelCheckpoint(monitor='mae_loss',
    #                                       dirpath=paths_manager.get_checkpoints_path(appliance, model_name),
    #                                       filename=f'model-{model.model.architecture_name}' + '-{epoch:02d}-{val_loss:.6f}',
    #                                       save_top_k=1,
    #                                       mode='min')
    #
    # checkpoint_callback = ModelCheckpoint()
    trainer = pl.Trainer(default_root_dir=paths_manager.get_report_path(appliance, model_name),
                         gpus=1, max_epochs=epochs,
                         auto_lr_find=True)
    # trainer.tune(model, train_loader, val_loader)
    if val_loader:
        trainer.fit(model, train_loader, val_loader)
        valdata = []
        for x, y in val_loader:
            valdata.extend(y.numpy())
        valdata = np.array(valdata)
        model.set_ground(valdata)
    else:
        trainer.fit(model, train_loader)
        valdata = []
        for x, y in train_loader:
            valdata.extend(y.numpy())
        valdata = np.array(valdata)
        model.set_ground(valdata)

    checkpointpath = paths_manager.get_checkpoints_path(appliance, model_name)
    trainer.save_checkpoint(os.path.join(checkpointpath, "last.ckpt"))

    # new_model = model.load_from_checkpoint(checkpoint_path=os.path.join(checkpointpath, "last.ckpt"))


    res = trainer.test(model, test_dataloaders=train_loader)
    print(res)
    test_result = model.get_res()
    results = test_result['metrics']
    preds = test_result['preds']
    save_report(root_dir=paths_manager.get_report_path(appliance, model_name), results=results, preds=preds,
                ground=valdata)

    torch.save(model.model.state_dict(), paths_manager.get_saved_models_path())
