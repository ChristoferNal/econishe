import torch
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F

from nilmlab.NILM_metrics import NILMMetrics

# pl.seed_everything(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)

ON_THRESHOLDS = {'dishwasher1'    : 10,
                 'refrigerator1'  : 50,
                 'kettle'         : 2000,
                 'microwave1'     : 200,
                 'washing machine': 20,
                 'furnace1'       : 10,
                 'drye1'          : 10,
                 'air1'           : 10,
                 'bedroom1'       : 10,
                 'bedroom2'       : 10}


class ClassicTrainingTools(pl.LightningModule):

    def __init__(self, model, eval_params, learning_rate=0.001):
        """
        Inputs:
            model_name - Name of the model to run. Used for creating the model (see function below)
            model_hparams - Hyperparameters for the model, as dictionary.
        """
        super().__init__()
        # self.learning_rate = learning_rate
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        # Create model
        self.model = model
        self.save_hyperparameters()

        self.eval_params = eval_params
        self.model_name = self.model.architecture_name

        self.final_preds = np.array([])
        self.results = {}

    def forward(self, x):
        # Forward function that is run when visualizing the graph
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters())

    def training_step(self, batch, batch_idx):
        # x must be in shape [batch_size, 1, window_size]
        x, y = batch
        outputs = self(x)
        # loss = F.mse_loss(outputs.squeeze(1), y)
        loss = self._compute_loss(outputs, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def train_epoch_end(self, outputs):
        # outputs is a list of whatever you returned in `training_step`
        train_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log("loss", train_loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        # loss = F.mse_loss(outputs.squeeze(), y.squeeze())
        loss = self._compute_loss(outputs, y)
        preds_batch = outputs.squeeze().cpu().numpy()
        self.final_preds = np.append(self.final_preds, preds_batch)
        return {'test_loss': loss}
        # return {'test_loss': loss, 'metrics': self._metrics(test=True)}

    def test_epoch_end(self, outputs):
        # outputs is a list of whatever you returned in `test_step`
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        tensorboard_logs = {'test_avg_loss': avg_loss}
        res = self._metrics()
        print('#### model name: {} ####'.format(res['model']))
        print('metrics: {}'.format(res['metrics']))

        self.log("test_test_avg_loss", avg_loss, 'log', tensorboard_logs)
        return res

    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        # loss = F.mse_loss(outputs.squeeze(1), y)
        loss = self._compute_loss(outputs, y)
        self.log('val_loss', loss, prog_bar=True)

    def _compute_loss(self, outputs, y):
        # loss = torchmetrics.functional.mean_absolute_error(outputs.squeeze(1), y)
        # loss = torchmetrics.functional.regression.mean_squared_log_error(outputs.squeeze(1), y)
        loss = F.mse_loss(outputs.squeeze(), y.squeeze())
        return loss

    def _metrics(self):
        device, mmax, groundtruth = self.eval_params['device'], \
                                    self.eval_params['mmax'], \
                                    self.eval_params['groundtruth']

        res = NILMMetrics(pred=self.final_preds,
                          ground=groundtruth,
                          mmax=mmax,
                          threshold=ON_THRESHOLDS[device])

        results = {'model'  : self.model_name,
                   'metrics': res,
                   'preds'  : self.final_preds, }
        self.set_res(results)
        self.final_preds = np.array([])
        return results

    def set_ground(self, ground):
        self.eval_params['groundtruth'] = ground

    def set_res(self, res):
        print("set_res")
        self.reset_res()
        self.results = res

    def reset_res(self):
        self.results = {}

    def get_res(self):
        print("get res")
        return self.results
