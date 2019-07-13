import json

from torch.utils.data import Dataset

from config import input_dims
from helpers.utils import preprocessing


class MyDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path) as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = preprocessing(self.data[index][0], input_dims)
        y = self.data[index][1]

        return x, y
