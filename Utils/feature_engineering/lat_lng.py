# 基于经纬度的特征工程:Part1
# 现在非常多的问题，例如道路交通管理，滴滴车定位接送用户等，经纬度是非常重要的地理信息。如何基于经纬度信息构建更好的特征对算法建模至关重要，本文，我们会介绍关于经纬度的几种非常重要的特征技巧。

# 此处我们介绍第一部分，我们将经纬度类比是坐标轴上的x轴,y轴，那么我们可以从数学的角度对其进行一些特征工程的构建。

# 1.1  两个经度/纬度的相减
# 使用相邻经度进行相减，相邻纬度相减，类似于经纬度的绝对变化特征

def lat_diff(lat1, lat2): 
    return lat1 - lat2

def lat_absdiff(lat1, lat2): 
    return abs(lat1 - lat2)

def lng_diff(lng1, lng2): 
    return lng1 - lng2

def lng_absdiff(lng1, lng2): 
    return abs(lng1 - lng2)


# 两个经度/纬度的相除
# 计算两个经度或者纬度的相除，减法如果是绝对特征，那么相除可以认为是相对特征

def lat_ratio(lat1, lat2): 
    return lat2 / lat1

def lgn_ratio(lgn1, lgn2): 
    return lgn2 / lgn1


# 经纬度相除
# 计算经纬度的相除的特征，类似于斜度计算

def lng_lat_ratio(lat1,lng1): 
    return lng1 / lat1

def lat_lng_ratio(lgn1, lgn2): 
    return lat1 / lgn1


# 两个经纬度的欧几里得距离计算
# 计算两个经纬度之间的欧几里得距离


def euclidean_distance(lat1, lng1, lat2, lng2): 
    return np.sqrt((lat1 - lat2) ** 2 + (lng1 - lng2) ** 2)


# 两个经纬度的Manhattan距离计算
# 计算两个经纬度之间的Manhattan距离

def manhattan_distance(lat1, lng1, lat2, lng2): 
    return abs(lat1 - lat2)  + abs(lng1 - lng2)


# 经纬度与斜边的比例特征
# 类似于三角形中两个直角边与斜边的比例信息

def lat_lng_hypotenuse_ratio(lat1, lng1): 
    hypotenuse = np.sqrt((lat1 ** 2 +  lng1 ** 2))
    return lat1 /  hypotenuse, lng1/  hypotenuse


# 基于经纬度的聚类1
# 基于经纬度进行聚类，将经纬度聚类的结果当做特征,例如常用的Kmeans。

# 此处我们用1表示,在第二部分我们会介绍升级版

from sklearn.cluster import MiniBatchKMeans
coords = np.vstack((train[['pickup_latitude', 'pickup_longitude']].values,
                    train[['dropoff_latitude', 'dropoff_longitude']].values,
                    test[['pickup_latitude', 'pickup_longitude']].values,
                    test[['dropoff_latitude', 'dropoff_longitude']].values))

sample_ind = np.random.permutation(len(coords))[:500000]
kmeans = MiniBatchKMeans(n_clusters=100, batch_size=10000).fit(coords[sample_ind])

train.loc[:, 'pickup_cluster'] = kmeans.predict(train[['pickup_latitude', 'pickup_longitude']])
train.loc[:, 'dropoff_cluster'] = kmeans.predict(train[['dropoff_latitude', 'dropoff_longitude']])
test.loc[:, 'pickup_cluster'] = kmeans.predict(test[['pickup_latitude', 'pickup_longitude']])
test.loc[:, 'dropoff_cluster'] = kmeans.predict(test[['dropoff_latitude', 'dropoff_longitude']])


# 特殊经纬度信息
# 例如第一个出现的经纬度可能是出发地点，最后一个现实的经纬度是终点站，那么这两个就是非常重要的特殊经纬度



# 距离某些特殊地点的经纬度距离
# 例如距离地铁的距离，距离汽车站的距离(距离可以是Manhattan距离等等)

def manhattan_distance_tostation(lat1, lng1, st_lat, st_lng): 
    return abs(lat1 - st_lat)  + abs(lng1 - st_lng)

def euclidean_distance_tostation(lat1, lng1, st_lat, st_lng): 
    return np.sqrt((lat1 - st_lat) ** 2 + (lng1 - st_lng) ** 2)