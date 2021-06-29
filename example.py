import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from datasources.torchdataset import PowerDataset, COLUMN_DATE, COLUMN_MAINS
from nilmmodels.disaggregators import SAEDDisagregator

appliance = 'refrigerator1'
model_name = 'SAED'
window = 50
house_path = 'data/house_7901.csv'
batch = 64

dataset = PowerDataset(path=house_path, device=appliance, window_size=window)
dataloader = DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=8)

disag = SAEDDisagregator(appliance, model_name, window)
cols = [COLUMN_DATE, COLUMN_MAINS, appliance]
data = pd.read_csv(house_path, usecols=cols)


for i, data in enumerate(dataloader):
    mains, meter = data
    predictions = disag.disaggregate(mains).detach().numpy().ravel()
    mains = mains.reshape(window, batch)[-1].detach().numpy().ravel()
    meter = meter.detach().numpy().ravel()

    plt.plot([x for x in range(i*batch, i*batch + batch)], mains, c='black')
    plt.plot([x for x in range(i*batch, i*batch + batch)], meter, c='blue')
    plt.plot([x for x in range(i*batch, i*batch + batch)], predictions, c='red')
    # plt.scatter([x for x in range(i*batch, i*batch + batch)], mains, s=1, c='black')
    # plt.scatter([x for x in range(i*batch, i*batch + batch)], meter, s=1, c='blue')
    # plt.scatter([x for x in range(i*batch, i*batch + batch)], predictions, s=1, c='red')
    plt.pause(0.05)

plt.show()