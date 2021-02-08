import util2 as u  # 匯入自訂工具模組

history_dict = {'loss': [0.8,0.7,0.4,0.2,0.1],   #} 2 個 key 的數據字典
                'acc':  [0.2,0.3,0.5,0.7,0.9]}   #}
u.plot(history_dict, ('loss','acc'),
       'Loss & Accuracy',
       ('Epochs', 'Loss & Acc'))

