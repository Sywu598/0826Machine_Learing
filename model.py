import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


X, y = make_regression(
    n_samples=300,       # 300 筆資料
    n_features=3,        # 只有 1 個特徵
    n_informative=3,     # 這個特徵就是有用的
    noise=2,            # 加一點雜訊，但不要太大
    random_state=123     # 因為random_state 有設  所以你會跟我一樣
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=123
)


linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

y_pred = linear_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse:.2f}')
print(f'R-squared: {r2:.2f}')

joblib.dump(linear_model, 'model/linear_regression_model.pkl')