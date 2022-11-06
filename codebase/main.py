from model.random_forest import RFModelFitter
from model.arima import ArimaModel
from data_prep.dataset import DatasetClass, RegressionDataset, ClassificationDataset
from constants import PREDICTION_WINDOW
from knapsack_solver import KnapsackSolver
from collections import OrderedDict
import warnings
warnings.filterwarnings('ignore')

data_class = DatasetClass()
ticker_list = data_class._get_tickers()

arima_dataset = RegressionDataset()
rf_dataset    = ClassificationDataset()

rf_fitter = RFModelFitter()
arima = ArimaModel()

# print("ticker list", ticker_list)
# for ticker in ticker_list:
#     rf_fitter.fit_ticker(ticker=ticker)

profit_ticker_sumamry = OrderedDict()

for ticker in ticker_list:
    ticker_dict = rf_fitter._load_rf_model(ticker_name=ticker)
    
    rf_feat_last_sample = rf_dataset._get_last_sample_features(ticker_name=ticker)
    rf_model = ticker_dict['rf_model']
    buy_sell_pred = rf_model.predict(X_test=rf_feat_last_sample)[0]

    if buy_sell_pred == 0:
        continue

    price_history = arima_dataset._get_full_dataset(ticker_name=ticker)
    # print("price_history", price_history)

    buy_price = price_history[-1]
    pred_prices = arima.predict_price_multiple_days_autoregressive(price_hist_list=price_history, N=PREDICTION_WINDOW)
    estimated_profit_win_end = pred_prices[-1] - buy_price # profit if we sold it after PREDICTION_WINDOW days (PREDICTION_WINDOW=5 in this case)
    estimated_profit_win_start = pred_prices[0] - buy_price # profit if we sold it the very next day

    # If RF said BUY, but ARIMA says there will be net loss, we ignore this Stock
    if estimated_profit_win_end < 0:
        continue

    profit_ticker_sumamry[ticker] = {'buy_price': buy_price, 'est_profit_win_end': estimated_profit_win_end, 
                                     'est_profit_win_start': estimated_profit_win_start,'rf_buy_sell_pred': buy_sell_pred}

print("Profitable Stocks", len(profit_ticker_sumamry), profit_ticker_sumamry.keys())

#####
# Now, from the profitable stocks, find optimal stocks to buy with the 1000 USD limit
#####

ind2key = {}
key2ind = {}
for index, key in enumerate(profit_ticker_sumamry.keys()):
    ind2key[index] = key
    key2ind[key] = index

weight_arr = [] # essentially buying price
profit_arr = [] # essentially Net Profit after PREDICTION_WINDOW days
MAX_ALLOWABLE_WEIGHT = 1000 # Given in question -> 1000 USD to play with
N = len(profit_ticker_sumamry)

SCALE_FACTOR = 1000 # To remove fractional profits

for i in range(0, N):
    ticker_name = ind2key[i]
    buy_cost = profit_ticker_sumamry[ticker_name]['buy_price']
    est_profit = profit_ticker_sumamry[ticker_name]['est_profit_win_end']

    weight_arr.append(round(buy_cost))
    profit_arr.append(round(est_profit*SCALE_FACTOR))

knapsack_cls = KnapsackSolver(N=N, weight_arr=weight_arr, profit_arr=profit_arr, max_allowable_weight=MAX_ALLOWABLE_WEIGHT)
knapsack_cls.solve()

stock_buy_list = knapsack_cls.dp_state_tracker[-1][-1]

# print("stock_buy_list", stock_buy_list)
stock_names_buy_list = [ind2key[key] for key in stock_buy_list]
print("stock_names_buy_list", stock_names_buy_list)

tot_buy_price=0
expected_profit=0
for ticker in stock_names_buy_list:
    tot_buy_price += profit_ticker_sumamry[ticker]['buy_price']
    expected_profit += profit_ticker_sumamry[ticker]['est_profit_win_end']


print("Total Investment:", tot_buy_price, "Estimated Profit", expected_profit)


