
from tensorflow.keras.utils import get_file

try:
    path = get_file('babi-tasks-v1-2.tar.gz',     # 要儲存的檔名
                    origin='https://s3.amazonaws.com/text-datasets/' # 下載的網址
                           'babi_tasks_1-20_v1-2.tar.gz')
except:
    print('下載資料集錯誤！')
    raise  # 再次丟出例外以結束程式

print('已下載到：' + path)

###############################################################
# 由傳入的檔案物件讀取資料, 傳回解析好的故事、問題、答案共 3 個串列
# 傳回的每個串列均內含 1000 個字串 (如果是開啟 10k 的任務檔則有 10000 個字串)
def get_sqa(f):           # f 為已開啟的檔案物件
    s, q, a = [], [], []  # 用來儲存故事、問題、答案的串列
    ts = []               # 用來暫存每個問題之前的所有敘述

    for line in f:  # 走訪檔案中的每一行字串 (可能是敘述或問題)

        line = line.decode('utf-8').strip()  # 由壓縮檔讀出的一行, 要先以 utf-8 解碼, 然後去除前後空白
        nid, line = line.split(' ', 1)       # 以空白字元將字串切割為編號和文字描述 (1 表示只切一次就好)
        if int(nid) == 1:    # 如果編號為 1, 表示為新故事的開始
            ts = []          # 清空暫存故事串列

        if '\t' in line:  # 如果字串中有 Tab 字元, 表示其為問題 (而不是敘述)
            tq, ta, tsup = line.split('\t')  #用 Tab 字元切割為問題、答案、及支持答案的敘述編號
            q.append(tq)  # 將問題 (字串) 加到問題串列中
            a.append(ta)   # 將答案 (字串) 加到答案串列中

            sup = ''      # 用來串接支持答案的敘述
            for i in tsup.split():     # 將支持答案的敘述編號以空白切割
                sup += ts[int(i) - 1]     # 將每個支持答案的敘述串接起來 (ts 陣列索引由 0 算起所以要減 1)
            sup = sup.replace('.', ' . ') # 將句點前後加空白, 使成為單字 (未來會以空白字元做斷字)
            s.append(sup)                 # 將串接好的支持敘述 (字串) 加到故事串列中

            ts.append('')   # 將空字串加到暫存故事串列中佔個位子, 以維持串列索引與敘述編號的對應關係

        else:    # 若沒有 Tab 則為敘述 (而非問題)
            ts.append(line) # 將敘述加到暫存故事串列中

    return s, q, a    #傳回解析的結果 (故事、問題、答案 3 個串列)

###############################################################

import tarfile   # 匯入套件

zpath = 'tasks_1-20_v1-2/en/qa2_two-supporting-facts'  # QA2 檔案在壓縮檔中的路徑及檔名前半段
with tarfile.open(path) as tar:     # 開啟壓縮檔
    s,  q,  a  = get_sqa(tar.extractfile(zpath + '_train.txt'))  # 讀取並解析壓縮檔中的 QA2 訓練檔
    ts, tq, ta = get_sqa(tar.extractfile(zpath + '_test.txt' ))  # 讀取並解析壓縮檔中的 QA2 測試檔

print('共', len(s), '筆訓練資料,', len(ts), '筆測試資料')      # 顯示傳回資料集的資訊
print('第 0、1、2 筆訓練故事/問題/答案：')
for i in range(3):                             # 顯示前 3 筆訓練故事/問題/答案
    print(f'{s[i]}/{q[i]}/{a[i]}')
print('第 0、1、2 筆測試故事/問題/答案：')
for i in range(3):                             # 顯示前 3 筆測試故事/問題/答案
    print(f'{ts[i]}/{tq[i]}/{ta[i]}')

###############################################################

from tensorflow.keras.preprocessing.text import Tokenizer

print('\n預處理：編製字典並將資料向量化')
tok = Tokenizer(filters='?')      # 建立 Tokenizer 物件, 要濾掉問號 (句點則保留)
tok.fit_on_texts(s+q+a+ts+tq+ta)  # 用全部的故事、問題、與答案來編製字典
vocab_size = len(tok.word_index) + 1  # 加上保留的索引 0
print(f'字典中有 {vocab_size} 個單字 (包含保留的索引 0)')
print('字典中每個單字出現的次數：\n', tok.word_counts)
print('字典中每個單字在字典中的索引：\n', tok.word_index)

xs = tok.texts_to_sequences(s)  # 將故事資料向量化
xq = tok.texts_to_sequences(q)  # 將問題資料向量化
y  = tok.texts_to_matrix(a)     # 將答案直接做 one-hot 編碼

txs = tok.texts_to_sequences(ts) #} 以同樣方式處理測試資料
txq = tok.texts_to_sequences(tq) #}
ty  = tok.texts_to_matrix(ta)    #}

print('\n將訓練資料向量化後的前 2 筆故事、問題、答案：')
print('xs =', xs[:2])
print('xq =', xq[:2])
print('y  =',  y[:2])

maxlen_s = max(map(len, xs + txs))  # 計算故事資料的最大長度
maxlen_q = max(map(len, xq + txq))  # 計算問題資料的最大長度

print('故事、問題資料的最大長度：', maxlen_s, '及', maxlen_q)

from tensorflow.keras.preprocessing.sequence import pad_sequences

xs  = pad_sequences(xs,  maxlen_s)  #} 將故事資料填充到一樣長
txs = pad_sequences(txs, maxlen_s)  #}
xq  = pad_sequences(xq,  maxlen_q)  #} 將問題資料填充到一樣長
txq = pad_sequences(txq, maxlen_q)  #}

print('預處理好的訓練資料的 shape：')
print('xs.shape = {}'.format(xs.shape))
print('xq.shape = {}'.format(xq.shape))
print('y.shape  = {}'.format(y.shape))

###########################################

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, concatenate, Dense

print('\n建立、編譯、訓練模型...')
xs_in = Input(shape=(maxlen_s,), dtype='int32')
xs_encoded = Embedding(vocab_size, 50)(xs_in)
xs_encoded = LSTM(100)(xs_encoded)

xq_in = Input(shape=(maxlen_q,), dtype='int32')
xq_encoded = Embedding(vocab_size, 50)(xq_in)
xq_encoded = LSTM(100)(xq_encoded)

out = concatenate([xs_encoded, xq_encoded])
out = Dense(vocab_size, activation='softmax')(out)

model = Model([xs_in, xq_in], out)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['acc'])

###########################################

history = model.fit([xs, xq], y, validation_split=0.05,
                    batch_size=32, epochs=20, verbose=2)

import util5 as u   # 匯入自訂模組 (plot() 用法參見 2-1 節的最後單元)

u.plot(history.history, ('acc', 'val_acc'),   #←繪製訓練及驗證的 mae 歷史線圖
       'Training & Validation Acc', ('Epoch','Acc'))

loss, acc = model.evaluate([txs, txq], ty, verbose=0)
print(f'\n評量成效：loss / Acc = {loss:.4f} / {acc:.4f}')