from tensorflow.keras import metrics
from tensorflow.keras import backend as K    #←匯入 Keras 的後端函式庫介面

y_pred = [[0.5,0.4,0.1],[0.3,0.2,0.5],[0.1,0.2,0.7]] #←建立預測值的張量
y_true = [[ 1,  0,  0 ],[ 1,  0,  0 ],[ 1,  0,  0 ]] #←建立標籤值的張量
for i in range(1, 4):
    ret = K.eval(metrics.top_k_categorical_accuracy(y_true, y_pred, k=i))
    acc = sum(ret)/len(ret)   #←計算平均值 (ret 為一個串列)
    print(f'top {i} 的傳回值：{ret}, 準確率 = {acc:.2f}')
