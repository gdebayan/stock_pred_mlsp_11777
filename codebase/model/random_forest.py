from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from data_prep.dataset import ClassificationDataset
from constants import TRAIN_SPLIT, RF_MODEL_SAVE_PATH
import pickle
import os

class RandomForest:

    def __init__(self) -> None:
        self.model = None

    def fit(self, X_train, y_train):
        rf = RandomForestClassifier()
        params_rf = {'n_estimators': [110, 130, 140, 150, 160, 180, 200]}
        rf_gs = GridSearchCV(rf, params_rf, cv=5)
        rf_gs.fit(X_train, y_train)

        self.model = rf_gs.best_estimator_
        return self.model

    def predict(self, X_test, y_test=None):
        assert self.model is not None
        prediction = self.model.predict(X_test)

        if not y_test:
            return prediction

        acc = accuracy_score(y_test, prediction)
        f1  = f1_score(y_test, prediction)
        # print(classification_report(y_test, prediction))
        # print(confusion_matrix(y_test, prediction))

        return prediction, acc, f1


class RFModelFitter:

    def __init__(self) -> None:
        self.save_path = RF_MODEL_SAVE_PATH
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

    def _save_pkl(self, var, file_name):
        with open(file_name, 'wb') as handle:
            pickle.dump(var, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_pkl(self, file_name):
        with open(file_name, 'rb') as handle:
            var = pickle.load(handle)
            return var

    def _load_rf_model(self, ticker_name):
        save_path = f"{self.save_path}/{ticker_name}.pkl"
        return self._load_pkl(save_path)

    def fit_ticker(self, ticker):

        data = ClassificationDataset()
        data_df = data._generate_train_test_data(ticker)

        rf_model = RandomForest()

        y = data_df['pred']
        features = [x for x in data_df.columns if x not in ['pred']]
        X = data_df[features]

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size= int(TRAIN_SPLIT * len(X)),shuffle=False)

        rf_model.fit(X_train=X_train, y_train=y_train)
        prediction, acc, f1 = rf_model.predict(X_test=X_test, y_test=y_test)
        
        save_dict = {'ticker': ticker, 'rf_model': rf_model, 'acc': acc, 'f1': f1}     
        print(f"Ticker: {ticker}, Save Dict: {save_dict}")   
        save_path = f"{self.save_path}/{ticker}.pkl"

        self._save_pkl(var=save_dict, file_name=save_path)
