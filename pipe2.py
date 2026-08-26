# 把模型做在pipeline裡面   將來就可以一起存起來
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
# 最後我們再準備一個pipeline
# 把整個資料前處理
# 模型訓練的過程 包裹起來
# 然後整包送去訓練


model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression())
])

model_pipeline.fit(X_train, y_train)