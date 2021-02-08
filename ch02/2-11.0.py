
def mygen(fr=0, to=None):     # 定義可傳回生成器物件的函式
    while to==None or fr<to:  # 若未指定 to 則無上限
        yield fr    # 傳回動態產生的資料
        fr += 1     # 將計數值加 1

for i in mygen(2, 5):  # 走訪由 2 到 5 (不含 5) 的生成器物件
    print(i)

gen = mygen(0)    # 傳回一個由 0 開始且無上限的生成器物件
print('next(gen) =', next(gen))
for i in range(100000): next(gen) # 用 next() 讀取 100000 次
print('next(gen) =', next(gen))

