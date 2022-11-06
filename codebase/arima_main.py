from model.arima import ArimaModel
from data_prep.dataset import RegressionDataset
from sklearn.metrics import mean_squared_error
from constants import MAX_DAYS_HISTORY


dataset = RegressionDataset()
model   = ArimaModel()

TICKER_NAME = 'AAPL'

train_data, test_data = dataset._generate_train_test_data(ticker_name=TICKER_NAME)
train_data_recent = train_data[-MAX_DAYS_HISTORY:]

predictions = model.predict_price_multiple_days(price_hist_list=train_data_recent, future_price_list=test_data)

rmse = mean_squared_error(test_data, predictions)
# print('Testing Mean Squared Error: %.3f' % rmse)