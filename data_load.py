from pathlib import Path
import numpy as np  # linear algebra
import pandas as pd  # data_small processing, CSV file I/O (e.g. pd.read_csv)

class DataLoader:
    def __init__(self, base_data_dir=str()):
        self.train_dir = base_data_dir + 'train/'
        self.test_dir = base_data_dir + 'test/'
        self.val_dir = base_data_dir + 'val/'
    def get_df(self, path):
        lst = []
        normal_dir = Path(path + "NORMAL")
        pneumonia_dir = Path(path + "PNEUMONIA")
        normal_data = normal_dir.glob("*.jpeg")
        pneumonia_data = pneumonia_dir.glob("*.jpeg")
        for fname in normal_data:
            lst.append((fname, 0))
        for fname in pneumonia_data:
            lst.append((fname, 1))
        df = pd.DataFrame(lst, columns=['Image', 'Label'], index=None)
        s = np.arange(df.shape[0])
        np.random.shuffle(s)
        df = df.iloc[s, :].reset_index(drop=True)
        return df
    def get_train_data(self):
        return self.get_df(self.train_dir)
    def get_test_data(self):
        return self.get_df(self.test_dir)
    def get_val_data(self):
        return self.get_df(self.val_dir)
