import torch


class ModelCheckpoint:
    def __init__(self, file_path, monitor='val_loss', save_best_only=False, save_state_dict_only=False, period=1):
        self.file_path = file_path
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.save_state_dict_only = save_state_dict_only
        self.period = period
        self.last_best = None
        self.losses = []
        self.model = None

    def set_model(self, model):
        self.model = model

    def make_checkpoint(self, epoch, logs):
        self.losses.append(logs[self.monitor])

        if self.save_best_only is True and min(self.losses) == self.losses[-1]:
            joiner = self.file_path.rfind('/')
            filename = self.file_path[:joiner + 1] + "BEST_" + self.file_path[joiner + 1:]
            filename = filename.format(epoch=epoch, loss=logs['loss'], val_loss=logs['val_loss'])

            self.last_best = filename
            self.save_checkpoint(filename)

        elif self.save_best_only is False and epoch % self.period == 0:
            filename = self.file_path.format(epoch=epoch, loss=logs['loss'], val_loss=logs['val_loss'])
            self.save_checkpoint(filename)

    def save_checkpoint(self, filename):
        if self.save_state_dict_only:
            torch.save(self.model.state_dict(), filename)
        else:
            torch.save(self.model, filename)
