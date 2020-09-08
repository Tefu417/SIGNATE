# LightGBM
import lightgbm as lgb #LightGBM
model = lgb.LGBMClassifier() # モデルのインスタンスの作成
model.fit(X_train, y_train) # モデルの学習

# Acc : 0.8294117647058824
# logloss : 0.47890918438081387
# AUC : 0.9230011261261261

# 結果 0.8320099