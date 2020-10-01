# ランダムフォレスト
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(X_tr,Y_tr)

# ('adjusted_r2(train)     :0.9421148655903482',
#  'adjusted_r2(test)      :0.4683109395599869',
#  '平均誤差率(test)       :inf',
#  'MAE(test)              :0.21776470588235292',
#  'MedianAE(test)         :0.06000000000000005',
#  'RMSE(test)             :0.3561311721328335',
#  'RMSE(test) / MAE(test) :1.6353943614959943')

# adjusted_r2(X_train, Y_train, model) 0.938505741620255

# 結果 0.9156328