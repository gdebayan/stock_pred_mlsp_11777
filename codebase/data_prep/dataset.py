
import sys, os
import pandas as pd
from pathlib import Path
from finta import TA


sys.path.insert(0, '../')

from codebase import constants
DATASET_FOLDER = constants.DATASET_FOLDER
PREDICTION_WINDOW = constants.PREDICTION_WINDOW

class DatasetClass:

    def __init__(self) -> None:
        self.dataset_folder = DATASET_FOLDER
        self.train_split = constants.TRAIN_SPLIT
        assert os.path.exists(self.dataset_folder)

    def _get_tickers(self):
        tickers = os.listdir(self.dataset_folder)
        tickers = [x.replace('.csv', '') for x in tickers]
        return tickers

    def _generate_train_test_data(self, ticker_name):
        """
        Function Implemented in Child Class
        """
        raise NotImplementedError


class RegressionDataset(DatasetClass):

    def __init__(self, exp_smooth_alpha=0.65) -> None:
        super().__init__()
        self.col_list = ['Close']
        self.alpha = exp_smooth_alpha
        self.max_records = constants.MAX_DAYS_HISTORY


    def _generate_train_test_data(self, ticker_name):
        data_csv_path = f"{self.dataset_folder}/{ticker_name}.csv"
        data_df = pd.read_csv(data_csv_path)
        data_df = data_df[self.col_list].ewm(alpha=self.alpha).mean()
        data_df = data_df.iloc[-self.max_records:]

        train_data, test_data = data_df[0:int(len(data_df)*self.train_split)], data_df[int(len(data_df)*self.train_split):]
        return train_data['Close'].tolist(), test_data['Close'].tolist()

    def _get_full_dataset(self, ticker_name):
        data_csv_path = f"{self.dataset_folder}/{ticker_name}.csv"
        data_df = pd.read_csv(data_csv_path)
        data_df = data_df[self.col_list].ewm(alpha=self.alpha).mean()
        data_df = data_df.iloc[-self.max_records:]
        return data_df['Close'].tolist()


class ClassificationDataset(DatasetClass):

    def __init__(self, exp_smooth_alpha=0.65, window_len=PREDICTION_WINDOW) -> None:
        super().__init__()
        self.alpha = exp_smooth_alpha
        self.indicators = ['RSI', 'MACD', 'STOCH','ADL', 
                           'ATR', 'MOM', 'MFI', 'ROC', 
                           'OBV', 'CCI', 'EMV', 'VORTEX']
        self.max_records = constants.MAX_DAYS_HISTORY
        self.window_len = window_len


    def _generate_train_test_data(self, ticker_name):
        data_csv_path = f"{self.dataset_folder}/{ticker_name}.csv"
        data_df = pd.read_csv(data_csv_path)
        
        # Rename the Columns to format supported by the Finta Library.
        # The Finta Library is used to extract additional indicators.
        # (https://github.com/peerchemist/finta)
        data_df.rename(columns={"Close": 'close',
                                 "High": 'high',
                                 "Low": 'low',
                                 'Volume': 'volume',
                                 'Open': 'open'}, inplace=True)
        data_df = data_df.iloc[-self.max_records:]
        data_df = data_df.ewm(alpha=self.alpha).mean()
        data_df = self._get_indicator_data(data=data_df)
        data_df = self._produce_prediction(data=data_df, window=self.window_len)
        data_df = data_df.dropna() # Some indicators produce NaN values for the first few rows, we just remove them here

        return data_df

    def _get_last_sample_features(self, ticker_name):
        data_csv_path = f"{self.dataset_folder}/{ticker_name}.csv"
        data_df = pd.read_csv(data_csv_path)
        
        # Rename the Columns to format supported by the Finta Library.
        # The Finta Library is used to extract additional indicators.
        # (https://github.com/peerchemist/finta)
        data_df.rename(columns={"Close": 'close',
                                 "High": 'high',
                                 "Low": 'low',
                                 'Volume': 'volume',
                                 'Open': 'open'}, inplace=True)
        data_df = data_df.iloc[-self.max_records:]
        data_df = data_df.ewm(alpha=self.alpha).mean()
        data_df = self._get_indicator_data(data=data_df)
        data_df = self._produce_prediction(data=data_df, window=self.window_len)

        last_sample = data_df.iloc[-1]

        feature_cols = [x for x in data_df.columns if x not in ['pred']]
        X = data_df[feature_cols]
        
        last_sample_feat = X.iloc[-1].to_numpy()

        return last_sample_feat.reshape((1,-1))

    def _get_indicator_data(self, data):

        """
        Function that uses the finta API to calculate technical indicators used as the features
        :return:
        """

        for indicator in self.indicators:
            ind_data = eval('TA.' + indicator + '(data)')
            if not isinstance(ind_data, pd.DataFrame):
                ind_data = ind_data.to_frame()
            data = data.merge(ind_data, left_index=True, right_index=True)
        data.rename(columns={"14 period EMV.": '14 period EMV'}, inplace=True)

        # Also calculate moving averages for features
        data['ema50'] = data['close'] / data['close'].ewm(50).mean()
        data['ema21'] = data['close'] / data['close'].ewm(21).mean()
        data['ema15'] = data['close'] / data['close'].ewm(14).mean()
        data['ema5'] = data['close'] / data['close'].ewm(5).mean()

        # Instead of using the actual volume value (which changes over time), we normalize it with a moving volume average
        data['normVol'] = data['volume'] / data['volume'].ewm(5).mean()

        # Remove columns that won't be used as features
        del (data['open'])
        del (data['high'])
        del (data['low'])
        del (data['volume'])
        del (data['Adj Close'])
        
        return data


    def _produce_prediction(self, data, window):
        """
        Function that produces the 'truth' values
        At a given row, it looks 'window' rows ahead to see if the price increased (1) or decreased (0)
        :param window: number of days, or rows to look ahead to see what the price did
        """
        prediction = (data.shift(-window)['close'] >= data['close'])
        prediction = prediction.iloc[:-window]
        data['pred'] = prediction.astype(int)
        return data