# The normalize function is required to normalize the data for the neural network.

def normalize(X_train, X_valid, X_test, normalize_opt, excluded_feat):
    feats = [f for f in X_train.columns if f not in excluded_feat]
    if normalize_opt != None:
        if normalize_opt == 'min_max':
            scaler = preprocessing.MinMaxScaler()
        elif normalize_opt == 'robust':
            scaler = preprocessing.RobustScaler()
        elif normalize_opt == 'standard':
            scaler = preprocessing.StandardScaler()
        elif normalize_opt == 'max_abs':
            scaler = preprocessing.MaxAbsScaler()
        scaler = scaler.fit(X_train[feats])
        X_train[feats] = scaler.transform(X_train[feats])
        X_valid[feats] = scaler.transform(X_valid[feats])
        X_test[feats] = scaler.transform(X_test[feats])
    return X_train, X_valid, X_test