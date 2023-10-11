import numpy as np

def normalizer(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # norm_data = numerator / (denominator + 1e-7)
    norm_data = numerator / denominator
    return norm_data, np.min(data, 0), np.max(data, 0)

def slicing_window(data, n_in):
    list_of_features = []

    for i in range(len(data)-n_in+1):
        arr_features = data[i:(i+n_in), :]
        list_of_features.append(arr_features)

    return np.array(list_of_features)

def batch_generator_with_time(data, time, batch_size):
    no = len(data)
    idx = np.random.permutation(no)
    train_idx = idx[:batch_size]         
    X_mb = list(data[i] for i in train_idx)
    T_mb = list(time[i] for i in train_idx)
    return X_mb, T_mb

def random_generator(batch_size, z_dim, T_mb, max_seq_len):
    Z_mb = list()
    for i in range(batch_size):
        temp = np.zeros([max_seq_len, z_dim])
        temp_Z = np.random.uniform(0., 1, [T_mb[i], z_dim])
        temp[:T_mb[i],:] = temp_Z
        Z_mb.append(temp_Z)
    return Z_mb

