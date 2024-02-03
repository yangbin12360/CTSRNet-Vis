import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import NMF
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from tqdm import tqdm
import subprocess
import umap
from scipy.spatial.distance import cdist
import torch.nn.functional as F
import torch.nn as nn
import torch
import pickle
import json
import time
import pandas as pd
import numpy as np
import collections
import struct
import os
os.environ["OMP_NUM_THREADS"] = '1'


# from utils.multi_process import multi_process

# TODO: gmm cluster

POLLUTANTS = ["RSP",
              "As",
              "Be",
              "Cd",
              "Ni",
              "Pb",
              "Cr",
              "Hg",
              "Al",
              "Mn",
              "Fe",
              "Ca",
              "Mg",
              "V",
              "Zn",
              "Ba",
              "Cu",
              "Se",
              "Na+",
              "NH4+",
              "K+",
              "Cl-",
              "Br-",
              "NO3-",
              "SO4=",
              "TC"]

POLLUTANTS_25 = ["As",
                 "Be",
                 "Cd",
                 "Ni",
                 "Pb",
                 "Cr",
                 "Hg",
                 "Al",
                 "Mn",
                 "Fe",
                 "Ca",
                 "Mg",
                 "V",
                 "Zn",
                 "Ba",
                 "Cu",
                 "Se",
                 "Na+",
                 "NH4+",
                 "K+",
                 "Cl-",
                 "Br-",
                 "NO3-",
                 "SO4=",
                 "TC"]

# SENSORS = ["觀塘",
#            "元朗",
#            "深水埗",
#            "中西區",
#            "荃灣",
#            "葵涌",
#            "東涌",
#            "旺角",
#            "屯門",
#            "將軍澳"]

SENSORS = ["中西區",
  "元朗",
  "將軍澳",
  "屯門",
  "旺角",
  "東涌",
  "深水埗",
  "荃灣",
  "葵涌",
  "觀塘",]


def load_UCR(dataset):
    train_file = os.path.join('./datasets/UCR',
                              dataset, dataset + "_TRAIN.tsv")
    test_file = os.path.join('./datasets/UCR',
                             dataset, dataset + "_TEST.tsv")
    train_df = pd.read_csv(train_file, sep="\t", header=None)
    test_df = pd.read_csv(test_file, sep="\t", header=None)

    train_array = np.array(train_df)
    test_array = np.array(test_df)

    # move the labels to {0, 1, ..., L-1}
    labels = np.unique(train_array[:, 0])
    transform = {}
    for i, l in enumerate(labels):
        transform[l] = i

    train = train_array[:, 1:].astype(np.float32)
    train_labels = np.vectorize(transform.get)(train_array[:, 0])
    test = test_array[:, 1:].astype(np.float32)
    test_labels = np.vectorize(transform.get)(test_array[:, 0])

    dim_series = train.shape[1]

    scaler = MinMaxScaler()
    train = scaler.fit_transform(train)
    test = scaler.fit_transform(test)
    return train, train_labels, test, test_labels

    # Normalization for non-normalized datasets
    # To keep the amplitude information, we do not normalize values over individual time series, but on the whole dataset
    # if dataset not in [
    #     'AllGestureWiimoteX',
    #     'AllGestureWiimoteY',
    #     'AllGestureWiimoteZ',
    #     'BME',
    #     'Chinatown',
    #     'Crop',
    #     'EOGHorizontalSignal',
    #     'EOGVerticalSignal',
    #     'Fungi',
    #     'GestureMidAirD1',
    #     'GestureMidAirD2',
    #     'GestureMidAirD3',
    #     'GesturePebbleZ1',
    #     'GesturePebbleZ2',
    #     'GunPointAgeSpan',
    #     'GunPointMaleVersusFemale',
    #     'GunPointOldVersusYoung',
    #     'HouseTwenty',
    #     'InsectEPGRegularTrain',
    #     'InsectEPGSmallTrain',
    #     'MelbournePedestrian',
    #     'PickupGestureWiimoteZ',
    #     'PigAirwayPressure',
    #     'PigArtPressure',
    #     'PigCVP',
    #     'PLAID',
    #     'PowerCons',
    #     'Rock',
    #     'SemgHandGenderCh2',
    #     'SemgHandMovementCh2',
    #     'SemgHandSubjectCh2',
    #     'ShakeGestureWiimoteZ',
    #     'SmoothSubspace',
    #     'UMD'
    # ]:
    #     # train[..., np.newaxis] shape: (num, length, 1)
    #     # labels [1, 2, ..., L]
    #     # train_labels shape: (num, )
    #     # return train[..., np.newaxis], train_labels, test[..., np.newaxis], test_labels, dim_series
    #     # return torch.tensor(train).view([-1, 1, dim_series]).to('cuda'), train_labels, torch.tensor(test).view([-1, 1, dim_series]).to('cuda'), test_labels
    #     return train, train_labels, test, test_labels

    # mean = np.nanmean(train)
    # std = np.nanstd(train)
    # train = (train - mean) / std
    # test = (test - mean) / std
    # return train, train_labels, test, test_labels


def load_HKAIR(mode, param):
    file = ''
    if mode == 'sensor':
        file = os.path.join('./data/hkair-by-sensor', param + '.json')
    elif mode == 'pollutant':
        file = os.path.join('./data/hkair-by-pollutant', param + '.json')
    with open(file, 'r', encoding='utf-8') as f:
        res = json.load(f)
    labels = list(res.keys())
    data = list(res.values())
    scaler = MinMaxScaler()
    data_p = scaler.fit_transform(np.array(data))
    return data_p, labels


def load_HKAIR_ALL(mode):
    file = ''
    if mode == 'sensor':
        print(1)
    elif mode == 'pollutant':
        file = './data/hk-air-by-pollutant-50/hkair-by-pollutant-50-na.json'
    with open(file, 'r', encoding='utf-8') as f:
        res = json.load(f)
    labels = list(res.keys())
    data = list(res.values())
    # scaler = MinMaxScaler()
    # data_p = scaler.fit_transform(np.array(data))
    return data, labels


def load_HKAIR_50(mode, param):
    if mode == 'pollutant':
        LABEL = ['中西區', '元朗', '將軍澳', '屯門', '旺角', '東涌', '深水埗', '荃灣', '葵涌', '觀塘']
        labels = []
        for label in LABEL:
            for i in range(5):
                labels.append(label)
    # TODO: mode==sensor

    file = './data/hk-air-by-pollutant-50/hkair-by-pollutant-50-na.json'
    with open(file, 'r') as f:
        data = json.load(f)

    data_p = np.array(data[param], dtype=np.float32)[:, :-1]
    # labels = np.array(list(data.keys()))

    scaler = MinMaxScaler()
    data_p = scaler.fit_transform(data_p)
    data_p = data_p.reshape([50, 61])
    return data_p, labels


def get_embedding_path_by_dataset(embedding_method, dataset_name):
    # exp_file_path = os.path.join("./default", embedding_method)
    # all_files = os.listdir(exp_file_path)
    # res_file_path = os.path.join(
    #     "./default", embedding_method, all_files[-1])
    # all_res_file = os.listdir(res_file_path)
    # for file in all_res_file:
    #     if file.endswith('npy'):
    #         embedding_path = os.path.join(res_file_path, file)
    #         break

    embedding_path = os.path.join(
        "./default", embedding_method, dataset_name + ".npy")

    # assert os.path.exists(embedding_path)

    return embedding_path


def embedData(name, model, model_name, dataset_name, embedding_filepath, data_size, batch_size, dim_series, device='cuda'):
    num_segments = int(data_size / batch_size)

    writer = FileContainer(embedding_filepath)

    try:
        with torch.no_grad():
            for segment in range(num_segments):
                if name == 'hkair':
                    batch, _ = load_HKAIR_50('pollutant', dataset_name)
                else:
                    batch, _, _, _ = load_UCR(dataset_name)
                batch = batch.reshape([-1, 1, dim_series])

                if model_name == 'seanet':
                    embedding = model.encode(torch.from_numpy(
                        batch).to(device)).detach().cpu().numpy()
                elif model_name == 'tsrnet':
                    _, _, _, embedding = model.forward(
                        torch.from_numpy(batch).to(device))
                    embedding = embedding.detach().cpu().numpy()

                np.save(embedding_filepath, embedding)
    finally:
        writer.close()


class FileContainer(object):
    def __init__(self, filename, binary=True):
        self.filename = filename
        self.binary = binary
        if self.binary:
            self.f = open(filename, "wb")
        else:
            self.f = open(filename, "w")

    def write(self, ts):
        if self.binary:
            s = struct.pack('f' * len(ts), *ts)
            self.f.write(s)
        else:
            self.f.write(" ".join(map(str, ts)) + "\n")

    def close(self):
        self.f.close()


def construct_pair(series, dim_series, per_subseries=0.1, theta=0.1):
    '''
    Args:
        series (2d array): the series to be paired
        dim_series (int): the length of original series
        per_subseries (float): the percentatge of the length of subseries
        theta (float): the percentage of the number of negative samples
    '''
    dim_subseries = int(dim_series * per_subseries)  # the length of subseries
    assert dim_subseries > 1, "sub series too short"

    sub_series = np.empty([0, dim_subseries])  # slice result

    # 1. slice subseries
    # subseries.shape: ((dim_series - dim_subseries + 1) * N, dim_subseries)
    for s in series:
        sub_s = np.lib.stride_tricks.sliding_window_view(
            s, window_shape=dim_subseries)
        sub_series = np.append(sub_series, sub_s, axis=0)
    # print(subseries.shape)

    # 2. compute kl_div matrix
    kl_list = []
    sub_tensor = torch.tensor(sub_series)
    num_sub_series = sub_series.shape[0]
    for i in range(num_sub_series):
        tensor_x = sub_tensor[i]
        x = F.log_softmax(tensor_x, dim=0)
        for j in range(i, num_sub_series):
            tensor_y = sub_tensor[j]
            y = F.softmax(tensor_y, dim=0)
            if j == i:
                continue
            else:
                kl = F.kl_div(x, y, reduction="mean")
                kl_list.append(float(kl))
    # print(sub_series.shape, len(kl_list), semi_to_full(kl_list))
    kl_matrix = _semi_to_full(kl_list)

    # 3. pair [O, P, N1, N2, ..., Nn] shape: (num_sub_series, 2+num_neg_sample, dim_subseries)
    pairs = []

    # the number of negative samples
    num_neg_sample = int(num_sub_series * theta)
    assert num_neg_sample > 0, "no satisfied negative samples, please higher theta"

    for index, anchor in enumerate(sub_series):
        neg_samples = []
        sorted_count_dict = _count_and_sort_dict(kl_matrix[index])
        for kl in sorted_count_dict:
            for s_index in kl[1]:
                if len(neg_samples) < num_neg_sample:
                    neg_samples.append(sub_series[s_index])
                else:
                    break
            if len(neg_samples) >= num_neg_sample:
                break

        pos_sample = sub_series[sorted_count_dict[-2][1][0]]

        neg_samples = np.array(neg_samples)
        pair = []
        pair.append(anchor)
        pair.append(pos_sample)
        for neg_sample in neg_samples:
            pair.append(neg_sample)

        pairs.append(pair)
    pairs = np.array(pairs)
    # print(pairs.shape)
    return pairs


def _semi_to_full(kl_list):
    # 生成对称矩阵 对角线上元素为0
    n = len(kl_list)
    n_matrix = int((1 + int((1 + 8 * n) ** 0.5)) / 2)
    semi_matrix = np.zeros((n_matrix, n_matrix))

    start_index = 0
    for row in range(n_matrix - 1):
        end_index = start_index + (n_matrix - 1 - row)
        semi_matrix[row, row+1:] = kl_list[start_index: end_index]
        start_index = end_index

    full_matrix = semi_matrix + semi_matrix.T
    return full_matrix


def _count_and_sort_dict(kl_row):
    count_dict = dict()
    for ki, kl in enumerate(kl_row):
        if kl in count_dict:
            count_dict[kl].append(ki)
        else:
            count_dict[kl] = [ki]

    sorted_count_dict = sorted(
        count_dict.items(), key=lambda x: x[0], reverse=True)  # 降序
    return sorted_count_dict


def cluster_embedding(method, dataset_name, embedding_path, dim_reduction, n_clusters=6):
    # data = np.load("./datasets/HK-RSP-NPY/HK_D.npy")  # [25, 50, 61]
    # idx = POLLUTANTS_25.index(dataset_name)
    # embeddings = data[idx, :, :]
    embeddings = np.load("./datasets/HK-RSP-EMBEDDING-NPY/" + dataset_name + ".npy")  # 50, 61, 320
    embeddings = embeddings.reshape([50, -1])

    labels = []
    for s in SENSORS:
        for i in range(5):
            labels.append(s + "_Y" + str(i))

    if dim_reduction == 'tsne':
        dr = TSNE(n_components=2, init='pca',
                  random_state=1, perplexity=5)
    elif dim_reduction == 'umap':
        dr = umap.UMAP(n_neighbors=10)

    dr_coord = np.array(dr.fit_transform(embeddings))

    if method == 'kmeans':
        k_means = KMeans(n_clusters=n_clusters, random_state=10)
        k_means.fit(embeddings)
        y = k_means.predict(embeddings)
        return dr_coord[:, 0], dr_coord[:, 1], y, labels
    else:
        raise ValueError('clustering method not supported.')


def nmf_transform(data, k):
    nmf = NMF(n_components=k)

    nmf.fit(data)  # train the model
    W = nmf.fit_transform(data)
    H = nmf.components_
    return W, H


def pkl_save(name, var):
    with open(name, 'wb') as f:
        pickle.dump(var, f)


def generate_confs():
    dataSummary = pd.read_csv('./datasets/UCRDataSummary.csv', header=0)
    res = {}

    for _, data in tqdm(dataSummary.iterrows()):
        res["dataset_name"] = data[2]
        res["num_class"] = data[5]
        res["dim_series"] = int(data[6])
        res["size_train"] = int(data[3] * 0.8)
        res["size_val"] = data[3] - int(data[3] * 0.8)
        res["size_test"] = data[4]

        des_path = './datasets/all_confs/' + res["dataset_name"] + '.json'
        with open(des_path, 'w') as f:
            json.dump(res, f)
        res = {}


def generate_hk_confs():
    # modes = ['sensor', 'pollutant']
    modes = ['pollutant']
    for mode in modes:
        root_path = './data/hkair-by-' + mode
        files = os.listdir(root_path)
        for file in files:
            res = {}
            dataset_name = file.split('.')[0]
            res_path = './datasets/confs_air_' + \
                mode + '_50/' + dataset_name + '-50.json'

            file_path = os.path.join(root_path, file)
            with open(file_path, 'r') as f:
                data = json.load(f)
            size = len(list(data.keys()))
            res["dataset_name"] = dataset_name
            # res["dim_series"] = len(list(data.values())[0])
            res["dim_series"] = 61
            res["size_train"] = 50
            # res["size_train"] = size
            res["size_val"] = 0
            res["size_test"] = size - res["size_train"]

            with open(res_path, 'w', encoding='utf-8') as ff:
                json.dump(res, ff)


def generate_dense():
    # 生成密度图的密度统计数据
    data_summary = pd.read_csv('./datasets/UCRDataSummary.csv', header=0)
    for _, line in tqdm(data_summary.iterrows()):
        dataset_name = line[2]

        counts = {}
        data, _, _, _ = load_UCR(dataset_name)
        data_ten = np.array(data * 10, dtype=float)
        data_ten = np.round(data_ten, decimals=0)
        data_ten_t = data_ten.T
        for time_stamp, d in enumerate(data_ten_t):
            counts[str(time_stamp)] = dict(collections.Counter(d))

        counts_arr = {}
        for time_stamp in counts:
            ts_count = []
            for i in range(11):
                if i in counts[time_stamp].keys():
                    ts_count.append(counts[time_stamp][i])
                else:
                    ts_count.append(0)
            counts_arr[time_stamp] = ts_count

        with open('./data/dense-statistic/' + dataset_name + '.json', 'w') as f:
            json.dump(counts_arr, f)


def generate_dense_hkair():
    # data_summary = pd.read_csv('./datasets/UCRDataSummary.csv', header=0)
    for pollutant in tqdm(POLLUTANTS):

        counts = {}
        data, _ = load_HKAIR_50("pollutant", pollutant)
        data_ten = np.array(data * 10, dtype=float)
        data_ten = np.round(data_ten, decimals=0)
        data_ten_t = data_ten.T
        for time_stamp, d in enumerate(data_ten_t):
            counts[str(time_stamp)] = dict(collections.Counter(d))

        counts_arr = {}
        for time_stamp in counts:
            ts_count = []
            for i in range(11):
                if i in counts[time_stamp].keys():
                    ts_count.append(counts[time_stamp][i])
                else:
                    ts_count.append(0)
            counts_arr[time_stamp] = ts_count

        with open('./data/dense-statistic-hkair/' + pollutant + '.json', 'w') as f:
            json.dump(counts_arr, f)


def preprocess_hk_rsp():
    # 1. 初步处理香港rsp数据
    sensor_file = './datasets/sensors.csv'
    sensors_df = pd.read_csv(sensor_file, sep=",", header=0)

    res_path = os.path.join(
        './datasets/HK-RSP-csv', 'hk_rsp.csv')

    root_path = './datasets/HK-RSP'
    files = os.listdir(root_path)

    all_df = pd.DataFrame(columns=['監測站', 'RSP', 'As', 'Be', 'Cd', 'Ni', 'P b', 'Cr', 'Hg', 'Al', 'Mn',
                                   'Fe', 'Ca', 'Mg', 'V', 'Zn', 'Ba', 'Cu', 'Se', 'Na+', 'NH4+', 'K+',
                                   'Cl-', 'Br-', 'NO3-', 'SO4=', 'TC', 'time', 'lon', 'lat'])
    for file in files:
        file_path = os.path.join(root_path, file)
        year = file.split('_')[2]

        # 处理时间
        df = pd.read_excel(file_path, sheet_name=year)
        df['月'] = df['月'].apply(lambda x: time.strftime(
            "%m", time.strptime(str(x), "%m")))
        df['日'] = df['日'].apply(lambda x: time.strftime(
            "%d", time.strptime(str(x), "%d")))
        df['time'] = df["年"].map(str) + df["月"] + df["日"]
        df.drop(['年', '月', '日'], axis=1, inplace=True)  # 删除原本 年、月、日 列，替换原有数据

        # 添加传感器经纬度
        df = df.merge(sensors_df, on=['監測站'], how='left')
        all_df = pd.concat([all_df, df])

    all_df.to_csv(res_path, sep=",", index=False)


def deal_missing():
    root_path = './datasets/HK-RSP-csv/hk_rsp.csv'
    df = pd.read_csv(root_path, header=0)


def preprocess_hk_rsp_by_sensor():
    # 按 传感器 组织数据
    file = './datasets/HK-RSP-csv/hk_rsp.csv'
    df = pd.read_csv(file, sep=",", header=0)

    for sensor in tqdm(SENSORS):
        res = {}
        res_path = os.path.join(
            './data/hkair-by-sensor', sensor + '.json')
        df_sensor = df[df['監測站'] == sensor]
        df_column = df_sensor.columns[1:28]
        arr_sensor = df_sensor.to_numpy().T[1:28]

        for ci, c in enumerate(df_column):
            res[c] = arr_sensor[ci].tolist()

        with open(res_path, 'w', encoding='utf-8') as f:
            json.dump(res, f)


def preprocess_hk_rsp_by_pollutant():
    # 按 污染物 组织数据
    root = './data/hkair-by-sensor/'
    files = os.listdir(root)

    for pollu in POLLUTANTS:
        res = {}
        res_path = os.path.join(
            './data/hkair-by-pollutant/', pollu+'.json')
        for file in files:
            file_path = os.path.join(root, file)
            sensor_name = file.split('.')[0]
            with open(file_path, 'r') as f:
                by_sensor = json.load(f)
            pollu_in_this_sensor = by_sensor[pollu]
            res[sensor_name] = pollu_in_this_sensor
        with open(res_path, 'w', encoding='utf-8') as f:
            json.dump(res, f)


def slice_hk_rsp_for_embed():
    # 将 hk 数据按年份切片, 得到 50 条数据用于训练
    sensor_file = './datasets/sensors.csv'
    sensors_df = pd.read_csv(sensor_file, sep=",", header=0)

    root_path = './datasets/HK-RSP'
    files = os.listdir(root_path)

    for file in files:
        file_path = os.path.join(root_path, file)
        year = file.split('_')[2]

        res_path = os.path.join(
            './data/hk-air-by-year', 'hk_rsp_' + year + '.csv')

        # 处理时间
        df = pd.read_excel(file_path, sheet_name=year)
        df['月'] = df['月'].apply(lambda x: time.strftime(
            "%m", time.strptime(str(x), "%m")))
        df['日'] = df['日'].apply(lambda x: time.strftime(
            "%d", time.strptime(str(x), "%d")))
        df['time'] = df["年"].map(str) + df["月"] + df["日"]
        df.drop(['年', '月', '日'], axis=1, inplace=True)  # 删除原本 年、月、日 列，替换原有数据

        # 添加传感器经纬度
        df = df.merge(sensors_df, on=['監測站'], how='left')

        df.to_csv(res_path, sep=",", index=False)


def slice_hk_rsp_for_embed_v2():
    root_path = './data/hkair-by-sensor'
    files = os.listdir(root_path)
    res_path = './data/hk-air-by-pollutant-50/hkair-by-pollutant-50.json'

    res = {}
    for pollutant in POLLUTANTS:
        pollu_arr = []
        for file in files:
            file_path = os.path.join(root_path, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            pollu_arr.append(data[pollutant])
        res[pollutant] = pollu_arr

    with open(res_path, 'w') as f:
        json.dump(res, f)


def prepare_rsp_npy():
    root_path = './datasets/HK-RSP-csv/hk_rsp_3060_sorted.csv'  # 手动删除了最后一个时间点的数据
    df = pd.read_csv(root_path, delimiter=',', header=0)

    res_path = './datasets/HK-RSP-NPY/'

    # df = df.sort_values(by=['監測站', 'time'])
    # df.to_csv("./datasets/HK-RSP-csv/hk_rsp_3060_sorted.csv", index=False, sep=',')

    rsp_data = np.array(list(df.iloc[:, [1]].values))
    data = np.array(list(df.iloc[:, 2: -3].values))  # 保留指标数据  (3050 * 25)

    data_at_timestamp = data.reshape([50, 61, 25]).swapaxes(0, 1)
    data_at_sensor = data.T.reshape([50, 25, 61])  # 某个监测站（在某年）的25种指标随时间的变化
    data_at_pollu = data_at_sensor.reshape(
        [25, 50, 61])  # 某个污染物在50个监测站随时间的变化 25 * 50 * 61
    np.save(os.path.join(res_path, 'HK_T.npy'), data_at_timestamp)
    np.save(os.path.join(res_path, 'HK_S.npy'), data_at_sensor)
    np.save(os.path.join(res_path, 'HK_D.npy'), data_at_pollu)
    np.save(os.path.join(res_path, 'HK_RSP.npy'), rsp_data)


# def many_clusters(dataset):
    # NUM_CLUSTER = 8  # TODO
    # res = multi_process(dataset, max_thread=8, max_cluster=8)

    # data = np.load("./datasets/HK-RSP-NPY/HK_D.npy")  # [25, 50, 61]
    # data = data[POLLUTANTS_25.index(dataset)]

    # silhouette = []
    # calinski = []
    # davis = []
    # all_cluster_num = []  # 每种聚类下每个簇内的样本数量
    # all_avg_dist = []

    # num_instance = data.shape[0]
    # dim_instance = data.shape[1]
    # for nc in range(2, NUM_CLUSTER):
    #     if cmethod == 'kmeans':
    #         k_means = KMeans(n_clusters=nc, random_state=10)
    #         y = k_means.fit_predict(data)
    #         y_T = np.array(y).reshape([num_instance, 1]).tolist()
    #     elif cmethod == 'gmm':
    #         gmm = GaussianMixture(n_components=nc, random_state=10)
    #         y = gmm.fit_predict(data)
    #         y_T = np.array(y).reshape([num_instance, 1]).tolist()

    #     # centroid
    #     centroids = k_means.cluster_centers_

    #     y_set = np.unique(y)

    #     columns = ['label']
    #     for i in range(dim_instance):
    #         columns.append(i)

    #     clusters_all = np.append(y_T, data, axis=1)
    #     clusters_all_df = pd.DataFrame(data=clusters_all, columns=columns)
    #     cluster_groups = clusters_all_df.groupby(['label'])

    #     avg_dist = []
    #     cluster_num = []  # 每个簇内的样本数量

    #     for yi, ys in enumerate(y_set):
    #         center = np.array(centroids[yi]).reshape([1, -1])
    #         cluster = np.array(
    #             list(cluster_groups.get_group(ys).values))[:, :-1]
    #         cluster_num.append(cluster.shape[0])

    #         avg_dist.append(cdist(cluster, center).mean())
    #     all_cluster_num.append(cluster_num)
    #     all_avg_dist.append(avg_dist)
    #     silhouette.append(silhouette_score(data, y))
    #     calinski.append(calinski_harabasz_score(data, y))
    #     davis.append(davies_bouldin_score(data, y))
    # return silhouette, calinski, davis, all_avg_dist, all_cluster_num


# def generate_PAA():

def select_center(points):
    # 给定一组数据点，确定中心点
    distances = cdist(points, points)
    sum_distances = np.sum(distances, axis=1)
    center_index = np.argmin(sum_distances)
    center_point = points[center_index]
    return center_point, center_index


def test_command():
    data_path = 'SyntheticControl'
    tl = 1

    command = "python3 ./utils/ccpca/ccpca/sample.py {} {}".format(
        data_path, tl)
    print(command)
    a = subprocess.check_output(command, shell=True)

    print("success", a.decode('utf-8'))


def tidy_rsp_embedding():
    # 整理ts2vec在hkair上的embedding数据, 用于可视化
    root_path = './datasets/rsp-embeddings'
    res_path = './datasets/HK-RSP-EMBEDDING-NPY'
    files = os.listdir(root_path)
    for file in tqdm(files):
        data_name = file.split('_')[0]
        file_path = os.path.join(root_path, file, 'embedding.npy')
        embedding = np.load(file_path)

        np.save(os.path.join(res_path, data_name+'.npy'), embedding)


def check_hk_embedding():
    data = np.load("./datasets/HK-RSP-EMBEDDING-NPY/As.npy")  # 50, 61, 320
    print(data.shape)


# if __name__ == '__main__':
#     prepare_rsp_npy()
#     check_hk_embedding()
#     tidy_rsp_embedding()
    # test_command()
    # many_clusters("pollutant", 'RSP')
    # generate_hk_confs()
    # load_HKAIR_50("pollutant", 'RSP')
    # slice_hk_rsp_for_embed_v2()
    # slice_hk_rsp_for_embed()
    # generate_hk_confs()
    # preprocess_hk_rsp()
    # preprocess_hk_rsp_by_sensor()
    # preprocess_hk_rsp_by_pollutant()
    # generate_dense()
    # generate_dense_hkair()
    # generate_confs()
    #     load_UCR('SyntheticControl')

    # series = np.ones((1, 60))
    # series = np.array([[-0.37793558, 1.2248643, 0.34387438, 0.32845403, -0.33760945, 1.026514, -1.330996, -0.65780029, 1.4460774, -0.84639658,
    #                     0.77620813, -0.25405199, 1.6652325, 0.62547205, -1.2921314, 0.95689775, -1.2161403, -0.58853578, 0.77698386, -0.60302893],
    #                    [0.64440621, 0.41326914, -0.86227849, -1.4973857, -0.42145781, -0.21421485, -1.2922114, 0.95621775, -1.2198403, -0.45853578,
    #                     0.77128386, -0.60302893, -0.30216806, -0.80695746, -0.37693558, 1.1248643, 0.39387438, 0.33945403, -0.39760945, 1.126514]])
    # num_series = series.shape[0]
    # dim_series = series.shape[1]
    # construct_pair(series=series, dim_series=dim_series,
    #                per_subseries=0.1, theta=0.1)

    # cluster_embedding('kmeans', 'SyntheticControl',
    #                   "./default/seanet/20230127162040/SyntheticControl-60-16.npy")
    # cluster_embedding('kmeans', 'SyntheticControl',
    #                   "./default/20221220162238/SyntheticControl-60-16.npy")
