def convert_str(x):
    '''
    对文本类型的列，通过字符串匹配（str.find）进行替换操作
    '''
    # 对工作年限进行转换
    x = str(x)
    if x.find('nan') > -1:
        return -1
    elif x.find("10+") > -1:  # 将"10＋years"转换成 11
        return 11
    elif x.find('< 1') > -1:  # 将"< 1 year"转换成 0
        return 0
    else:
        return int(re.sub("\D", "", x))  # 其余数据，去掉"years"并转换成整数