class EarlyStopping:
    def __init__(self, monitor='loss', patience=0, min_delta=0.0):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.count = 0
        self.best = None
        self.best_epoch = 0

    def check(self, epoch, logs):
        if self.best is None:
            self.best = logs[self.monitor]
            self.best_epoch = epoch
        else:
            current = logs[self.monitor]
            if (current - self.min_delta) < self.best:
                self.best = current
                self.best_epoch = epoch
                self.count = 0
            else:
                self.count += 1
                if self.count >= self.patience:
                    print('\nTraining stopped because of Early Stopping.')
                    print('According to the monitored values, the best epoch was at', str(self.best_epoch + 1), '.')
                    exit()