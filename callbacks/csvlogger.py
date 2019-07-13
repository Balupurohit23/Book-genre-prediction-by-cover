import csv
import os
from config import train_var, val_var


class CSVLogger:
    def __init__(self, file_path):
        fields = train_var + val_var
        self.file_path = file_path
        self.fieldnames = ['epoch'] + fields

        if not os.path.isdir(file_path[:file_path.rfind('/') + 1]):
            print("Directory doesn't exists. Log file will be saved in log folder after creating it.")
            os.mkdir('log')

        with open(self.file_path, 'w') as f:
            csv_logger = csv.DictWriter(f, fieldnames=self.fieldnames)
            csv_logger.writeheader()

    def update(self, epoch, logs):
        logs.update({"epoch": epoch})
        with open(self.file_path, 'a') as f:
            csv_logger = csv.DictWriter(f, fieldnames=self.fieldnames)
            csv_logger.writerow(dict(sorted(logs.items())))