from data_prep.dataset import ClassificationDataset
from model.random_forest import RandomForest
from sklearn.model_selection import train_test_split
from constants import TRAIN_SPLIT

data = ClassificationDataset()
data_df = data._generate_train_test_data('AAPL')
# print("data df tail", data_df.tail())
rf_model = RandomForest()

y = data_df['pred']
features = [x for x in data_df.columns if x not in ['pred']]
X = data_df[features]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size= int(TRAIN_SPLIT * len(X)),shuffle=False)

rf_model.fit(X_train=X_train, y_train=y_train)
prediction, acc, f1 = rf_model.predict(X_test=X_test, y_test=y_test)
print("acc", acc)