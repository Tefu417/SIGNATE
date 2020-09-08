# ランダムフォレスト
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(X_tr,Y_tr)

# ('adjusted_r2(train)     :0.9460319967564532',
#  'adjusted_r2(test)      :0.4828510468015187',
#  '平均誤差率(test)       :1.1269747882692045',
#  'MAE(test)              :0.20835294117647055',
#  'MedianAE(test)         :0.065',
#  'RMSE(test)             :0.3458323293158116',
#  'RMSE(test) / MAE(test) :1.6598389605784298')

# adjusted_r2(X_train, Y_train, model) 0.8588908867040684

# 結果 0.9157982