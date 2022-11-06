"""
This file implements the ARIMA Class.
The ARIMA model is a regression model, that outputs a price_t, given prices_1:t.
i.e. --> price(t) = arima_predict(price_list[0], price_list[1], ..... , price_list[t-1])
"""
from statsmodels.tsa.arima.model import ARIMA

class ArimaModel:

    def __init__(self, p=5, d=1, q=0) -> None:
        """
        p, d, q --> are parameters corresponding to the ARIMA model
        """
        self.arima = ARIMA
        self.p = p
        self.d = d
        self.q = q

    def predict(self, price_hist_list):
        """
        Predict price on day "t" based on prices from "0...t-1" days

        price_hist_list: Price from days "0 ... t-1"
        """
        model = self.arima(price_hist_list, order=(self.p, self.d, self.q))
        model_fit = model.fit()
        
        op = model_fit.forecast()
        pred_price =  op[0]
        return pred_price

    def predict_price_multiple_days(self, price_hist_list, future_price_list):
        """
        This method produces N predictions.
        N == Number of Entries (Days) in future_price_list

        Args:
            price_hist_list -> Price from days "0 ... t-1"
            future_price_list -> True Price Values for days "t, t+1, ......, t+N"
        """
        history = price_hist_list # Initialize the current History

        predictions = []

        N = len(future_price_list)
        for t in range(0, N):
            pred = self.predict(price_hist_list=history)
            predictions.append(pred)
            history.append(future_price_list[t])
        return predictions

    def predict_price_multiple_days_autoregressive(self, price_hist_list, N):
        """
        This method produces N predictions.

        Args:
            price_hist_list -> Price from days "0 ... t-1"
            N -> Number of days in future to predict price
        """
        history = price_hist_list # Initialize the current History

        predictions = []

        for t in range(0, N):
            pred = self.predict(price_hist_list=history)
            predictions.append(pred)
            history.append(pred)
        return predictions





