import sys

from tqdm import tqdm

from config import train_var, val_var


class PBar:
    def __init__(self, total=0):
        self.pbar = tqdm(total=total, file=sys.stdout, ncols=50)

    def set_desc(self, description):
        self.pbar.set_description(description)

    def set_batch_postfix(self, log, mode='training'):
        if mode == 'training':
            s = ''
            for var in train_var:
                s += self.add_string(var, log[var][-1])
            self.pbar.set_postfix_str(s)

        if mode == 'validation':
            s = ''
            for var in val_var:
                s += self.add_string(var[4:], log[var][-1])
            self.pbar.set_postfix_str(s)

    def set_epoch_postfix(self, log, mode='training'):
        if mode == 'training':
            s = ''
            for var in train_var:
                s += self.add_string(var, log[var])
            self.pbar.set_postfix_str(s)

        if mode == 'validation':
            s = ''
            for var in val_var:
                s+=self.add_string(var[4:], log[var])
            self.pbar.set_postfix_str(s)

    def add_string(self, key, value):
        return '   ' + key + '= %.4f' % value

    def update(self, num):
        self.pbar.update(num)

    def write(self, string):
        self.pbar.write(str(string))

    def close(self):
        self.pbar.close()