""" import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

class Dataset_StockKaggle(Dataset):
    def __init__(self, root_path="\AAPL", flag='train', size=None, 
                 file_name='AAPL.csv', features='S', target='Close', scale=True, 
                 timeenc=0, freq='d', train_ratio=0.7, val_ratio=0.15):
        """
""""root_path: Folder containing stock CSV files
        flag: 'train', 'val', or 'test'
        size: (seq_len, label_len, pred_len)
        file_name: Stock file to load (e.g., 'AAPL.csv')
        features: 'S' = univariate, 'M' = multivariate (not used yet)
        target: column to predict (e.g., 'Close')
        scale: whether to normalize data
        """
""""assert flag in ['train', 'val', 'test']
        self.set_type = flag
        self.seq_len, self.label_len, self.pred_len = size
        self.file_name = file_name
        self.target = target
        self.scale = scale
        self.root_path = root_path
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1 - train_ratio - val_ratio

        self.__read_data__()

    def __read_data__(self):
        df = pd.read_csv(os.path.join(self.root_path, self.file_name))
        #df['Date'] = pd.to_datetime(df['Date'])
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

        


        df = df.sort_values('Date')

        # Extract target column only
        data = df[[self.target]].values

        # Normalize
        if self.scale:
            self.scaler = StandardScaler()
            data = self.scaler.fit_transform(data)

        # Create windows
        self.data_x, self.data_y = [], []
        total_len = self.seq_len + self.pred_len
        for i in range(len(data) - total_len):
            seq_x = data[i : i + self.seq_len]
            seq_y = data[i + self.seq_len : i + total_len]
            self.data_x.append(seq_x)
            self.data_y.append(seq_y)

        self.data_x = np.array(self.data_x)
        self.data_y = np.array(self.data_y)

        # Split into train/val/test
        num_samples = len(self.data_x)
        train_end = int(self.train_ratio * num_samples)
        val_end = int((self.train_ratio + self.val_ratio) * num_samples)

        if self.set_type == 'train':
            self.data_x = self.data_x[:train_end]
            self.data_y = self.data_y[:train_end]
        elif self.set_type == 'val':
            self.data_x = self.data_x[train_end:val_end]
            self.data_y = self.data_y[train_end:val_end]
        else:  # test
            self.data_x = self.data_x[val_end:]
            self.data_y = self.data_y[val_end:]

    def __getitem__(self, index):
        # THUML expects (input, target, timestamp info, timestamp info)
        return self.data_x[index], self.data_y[index], 0, 0

    def __len__(self):
        return len(self.data_x) """

import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class Dataset_StockKaggle(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 file_name='AAPL.csv', features='S', target='Close',
                 scale=True, timeenc=0, freq='d', train_ratio=0.7, val_ratio=0.15):
        assert flag in ['train', 'val', 'test']
        self.seq_len, self.label_len, self.pred_len = size
        self.flag = flag
        self.root_path = root_path
        self.file_name = file_name
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

        self.__read_data__()

    def __read_data__(self):
        df = pd.read_csv(os.path.join(self.root_path, self.file_name))

        # ✅ Ensure 'Date' is parsed properly
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

        # ✅ Extract time features for time encoding
        df_stamp = df[['Date']].copy()
        df_stamp['month'] = df_stamp['Date'].dt.month
        df_stamp['day'] = df_stamp['Date'].dt.day
        df_stamp['weekday'] = df_stamp['Date'].dt.weekday
        # If using hourly data, include hour — otherwise, you may comment this out
        df_stamp['hour'] = df_stamp['Date'].dt.hour

        self.data_stamp = df_stamp.drop(['Date'], axis=1).values

        # ✅ Determine columns to use
        if self.features == 'M':
            self.cols = list(df.columns.drop(['Date']))
        elif self.features == 'S':
            self.cols = [self.target]
        else:
            raise ValueError("Invalid value for features. Use 'M' or 'S'.")

        self.data = df[self.cols].values

        # ✅ Scaling (optional)
        if self.scale:
            self.data_mean = self.data.mean(0)
            self.data_std = self.data.std(0)
            self.data = (self.data - self.data_mean) / self.data_std

        # ✅ Set data range
        total_len = len(self.data)
        border1s = {
            'train': 0,
            'val': int(total_len * self.train_ratio) - self.seq_len,
            'test': int(total_len * (self.train_ratio + self.val_ratio)) - self.seq_len
        }
        border2s = {
            'train': int(total_len * self.train_ratio),
            'val': int(total_len * (self.train_ratio + self.val_ratio)),
            'test': total_len
        }

        self.border1 = border1s[self.flag]
        self.border2 = border2s[self.flag]

    def __getitem__(self, index):
        s_begin = self.border1 + index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data[s_begin:s_end]
        seq_y = self.data[r_begin:r_end]

        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return self.border2 - self.border1 - self.seq_len - self.pred_len + 1
