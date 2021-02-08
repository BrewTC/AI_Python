# 以字典取得與還原模型結構的方法

from tensorflow.keras.models import Sequential
import util2 as u

model = u.mnist_model()  # 取得新建立並編譯好的模型

config = model.get_config()            # 以字典傳回模型結構
model = Sequential.from_config(config) # 載入模型結構

# 上一行若為 Function API 模型, 須改用 Model 來載入結構：
# from tensorflow.keras.models import Model
# model = Model.from_config(config)  # 載入 Functional API 模型結構

# 以 JSON 字串取得與還原模型結構的方法
from tensorflow.keras.models import model_from_json
json_string = model.to_json()         # 以 JSON 字串傳回模型結構
model = model_from_json(json_string)  # 載入模型結構

# 以 YAML 字串取得與還原模型結構的方法
from tensorflow.keras.models import model_from_yaml
yaml_string = model.to_yaml()         # 以 YAML 字串傳回模型結構
model = model_from_yaml(yaml_string)  # 載入模型結構

##################################################

# 用 JSON 格式儲存模型結構
from tensorflow.keras.models import model_from_json

json_string = model.to_json()
with open("model.config", "w") as text_file:
    text_file.write(json_string)

# 讀取 JSON 格式的模型結構並還原
from tensorflow.keras.models import model_from_json

with open("model.config", "r") as text_file:
    json_string = text_file.read()
    model = model_from_json(json_string)

##################################################

# 取得、還原權重參數
weights = model.get_weights()   # 取得權重參數
print(weights)
model.set_weights(weights)      # 還原權重參數

##################################################

# 儲存、載入權重參數
model.save_weights("model.weight")   # 儲存權重參數到 HDF5 格式的檔案中
model.load_weights("model.weight")   # 載入權重參數

