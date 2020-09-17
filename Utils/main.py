#读取指定文件并进行拼接
INPUT_PATH = '../input/'
df = taxiorder2019 = pd.concat([
    pd.read_csv(INPUT_PATH + x) for x in [
        'taxiOrder20190607.csv',
        'taxiOrder20190608.csv',
        'taxiOrder20190609.csv'
    ]
])
cal_taxi(df)
