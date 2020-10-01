# LightGBM
# 学習に使用するデータを設定
lgb_train = lgb.Dataset(X_tr, Y_tr)
lgb_eval = lgb.Dataset(X_te, Y_te, reference=lgb_train)

# LightGBM parameters
params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',      # 目的 : 2値分類
        'metric': {'binary_error'}, # 評価指標 : 誤り率(= 1-正答率)
        #他には 'binary_logloss','auc'など
        'learning_rate': 0.1,
        'num_leaves': 23,
        'min_data_in_leaf': 1,
        'num_iteration': 100,   #100回学習
        'verbose': 0
}

# モデルの学習
model = lgb.train(params,   # パラメータ
            train_set = lgb_train,    # トレーニングデータの指定
            valid_sets = lgb_eval,    # 検証データの指定
            early_stopping_rounds = 100   # 100回ごとに検証精度の改善を検討　→ 精度が改善しないなら学習を終了(過学習に陥るのを防ぐ)
               )

# テストデータの予測 (クラス1の予測確率(クラス1である確率)を返す)
y_pred_prob = model.predict(X_te)
# テストデータの予測
y_pred = np.where(y_pred_prob < 0.5, 0, 1) # 0.5より小さい場合0 ,そうでない場合1を返す

# ('adjusted_r2(train)     :0.9520453647766514',
#  'adjusted_r2(test)      :0.4351884939423748',
#  '平均誤差率(test)       :1.3909049333804138',
#  'MAE(test)              :0.2189915209889515',
#  'MedianAE(test)         :0.0545967950972115',
#  'RMSE(test)             :0.36141779548880854',
#  'RMSE(test) / MAE(test) :1.65037346586146')

# adjusted_r2(X_train, y_train, model) 0.8546583274176797

# 0.9064185