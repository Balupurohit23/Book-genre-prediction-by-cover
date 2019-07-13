from helpers.utils import average
from config import train_var, val_var


class History:
    def __init__(self):
        self.train_var = train_var
        self.val_var = val_var

        self.train_batch_logs = dict(zip(self.train_var, [[0.0]] * len(self.train_var)))
        self.val_batch_logs = dict(zip(self.val_var, [[0.0]] * len(self.val_var)))

        self.train_epoch_logs = dict(zip(self.train_var, [0.0] * len(self.train_batch_logs)))
        self.val_epoch_logs = dict(zip(self.val_var, [0.0] * len(self.val_var)))

    def reset_batch_logs(self):
        for key in list(self.train_batch_logs.keys()):
            self.train_batch_logs[key] = []

        for key in list(self.val_batch_logs.keys()):
            self.val_batch_logs[key] = []

    def reset_epoch_logs(self):
        for key in list(self.train_epoch_logs.keys()):
            self.train_epoch_logs[key] = []

        for key in list(self.val_epoch_logs.keys()):
            self.val_epoch_logs[key] = []

    def update_batch(self, logs, mode='training'):
        if mode =='training':
            i = 0
            for key in self.train_var:
                self.train_batch_logs[key].append(logs[i])
                i += 1

        if mode =='validation':
            i = 0
            for key in self.val_var:
                self.val_batch_logs[key].append(logs[i])
                i += 1

    def update_epoch(self, mode='training'):
        if mode == 'training':
            i = 0
            for key in self.train_var:
                self.train_epoch_logs[key] = average(self.train_batch_logs[self.train_var[i]])
                i += 1

        elif mode == 'validation':
            i = 0
            for key in self.val_var:
                self.val_epoch_logs[key] = average(self.val_batch_logs[self.val_var[i]])
                i += 1