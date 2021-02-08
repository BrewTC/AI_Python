import util2 as u  # 匯入自訂的工具模組
import time, math
from tensorflow.keras.callbacks import Callback   # 匯入回呼類別

class stop_at_98(Callback):  #←自訂回呼類別, 當驗證準確率到達98%或時間超過20分鐘即停止訓練
  def on_epoch_end(self, epoch, logs=None):
    val_acc = logs.get('val_acc')
    if val_acc>=0.98 or time.time()-start_time>1200:
      print(f'於第 {epoch+1:3} 週期到達 {val_acc:.3f}, '
            f'總訓練次數(總批次數)={(epoch+1)*math.ceil(60000/batch_size):6}, ' # 這裡的epoch是由0算起
            f'時間：{time.time()-start_time:7.2f} 秒')
      self.model.stop_training = True   #←停止訓練

def train(batch_size):  #←將訓練程式包在函式中, 以進行不同 batch_size 的訓練
  print(f'訓練 batch_size={batch_size:5} ', end='')
  model.fit(x_train, y_train, verbose=0, callbacks=[stop_at_98()],
            batch_size=batch_size, epochs=800,        #↑指定回呼物件
            validation_data = (x_test, y_test))  #←用測試資料做驗證

#####↓主程式↓#####
(x_train, x_test), (y_train, y_test) = u.mnist_data() #←載入預處理好的 MNIST 資料集
model = u.mnist_model()     #←取得新建立並編譯好的模型

wt = model.get_weights()    #←儲存模型的初始權重, 以供稍後重新訓練模型時使用

batch_size =  128  #←batch_size 由 1 開始測試
while batch_size < 60000*2:
  if batch_size > 60000: batch_size = 60000
  model.set_weights(wt)    #←重新載入初始權重
  start_time = time.time() #←記錄開始訓練的時間
  train(batch_size)        #←依 batch_size 訓練模型
  batch_size *= 2          #←將 batch_size 加倍