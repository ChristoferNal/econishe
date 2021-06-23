import pandas as pd


class PowerDataGenerator:

    @staticmethod
    def generate_data(path='../data/house_7901.csv', max_num_of_data=None):
        """
        Generates household power consumption data, including each appliance.
        If the original data finish, then it will start iterating from the beginning.
        The index that is returned is a global index which starts from 0 when the generator starts
        and continues until the generator is manually stopped or generates the maximum number of data.

        Example:
        for index, data in PowerDataGenerator.generate_data(path=..., max_num_of_data=100000):
            do something...
            microwave = data['microwave1']

        Columns that are used for house_7901.csv:
        dataid,localminute,air1,bedroom1,bedroom2,clotheswasher1,dishwasher1,disposal1,drye1,furnace1,grid,kitchenapp1,
        kitchenapp2,lights_plugs1,livingroom1,microwave1,refrigerator1,leg1v,leg2v
        """
        data = pd.read_csv(path)
        global_idx = 0
        stop = False
        print(data)
        while not stop:
            for _, row in data.iterrows():
                if max_num_of_data and global_idx == max_num_of_data:
                    stop = True
                    break
                global_idx += 1
                yield global_idx, row
