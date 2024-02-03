from flask import Flask, request
from flask_cors import CORS
import os
import ast
import numpy as np
import pandas as pd
import json
import collections
import subprocess
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

from utils.constants import TIMESTAMPS, POLLUTANTS

from utils.data import cluster_embedding, nmf_transform, load_UCR, get_embedding_path_by_dataset, load_HKAIR, load_HKAIR_50, load_HKAIR_ALL, select_center
from utils.multi_process import multi_process_cluster, multi_process_ft



app = Flask(__name__)
CORS(app, supports_credentials=True)

UCR_DATASETS = ["Adiac",
                "ArrowHead",
                "Beef",
                "BeetleFly",
                "BirdChicken",
                "Car",
                "CBF",
                "ChlorineConcentration",
                "CinCECGTorso",
                "Coffee",
                "Computers",
                "CricketX",
                "CricketY",
                "CricketZ",
                "DiatomSizeReduction",
                "DistalPhalanxOutlineAgeGroup",
                "DistalPhalanxOutlineCorrect",
                "DistalPhalanxTW",
                "Earthquakes",
                "ECG200",
                "ECG5000",
                "ECGFiveDays",
                "ElectricDevices",
                "FaceAll",
                "FaceFour",
                "FacesUCR",
                "FiftyWords",
                "Fish",
                "FordA",
                "FordB",
                "GunPoint",
                "Ham",
                "HandOutlines",
                "Haptics",
                "Herring",
                "InlineSkate",
                "InsectWingbeatSound",
                "ItalyPowerDemand",
                "LargeKitchenAppliances",
                "Lightning2",
                "Lightning7",
                "Mallat",
                "Meat",
                "MedicalImages",
                "MiddlePhalanxOutlineAgeGroup",
                "MiddlePhalanxOutlineCorrect",
                "MiddlePhalanxTW",
                "MoteStrain",
                "NonInvasiveFetalECGThorax1",
                "NonInvasiveFetalECGThorax2",
                "OliveOil",
                "OSULeaf",
                "PhalangesOutlinesCorrect",
                "Phoneme",
                "Plane",
                "ProximalPhalanxOutlineAgeGroup",
                "ProximalPhalanxOutlineCorrect",
                "ProximalPhalanxTW",
                "RefrigerationDevices",
                "ScreenType",
                "ShapeletSim",
                "ShapesAll",
                "SmallKitchenAppliances",
                "SonyAIBORobotSurface1",
                "SonyAIBORobotSurface2",
                "StarLightCurves",
                "Strawberry",
                "SwedishLeaf",
                "Symbols",
                "SyntheticControl",
                "ToeSegmentation1",
                "ToeSegmentation2",
                "Trace",
                "TwoLeadECG",
                "TwoPatterns",
                "UWaveGestureLibraryAll",
                "UWaveGestureLibraryX",
                "UWaveGestureLibraryY",
                "UWaveGestureLibraryZ",
                "Wafer",
                "Wine",
                "WordSynonyms",
                "Worms",
                "WormsTwoClass",
                "Yoga",
                "ACSF1",
                "AllGestureWiimoteX",
                "AllGestureWiimoteY",
                "AllGestureWiimoteZ",
                "BME",
                "Chinatown",
                "Crop",
                "DodgerLoopDay",
                "DodgerLoopGame",
                "DodgerLoopWeekend",
                "EOGHorizontalSignal",
                "EOGVerticalSignal",
                "EthanolLevel",
                "FreezerRegularTrain",
                "FreezerSmallTrain",
                "Fungi",
                "GestureMidAirD1",
                "GestureMidAirD2",
                "GestureMidAirD3",
                "GesturePebbleZ1",
                "GesturePebbleZ2",
                "GunPointAgeSpan",
                "GunPointMaleVersusFemale",
                "GunPointOldVersusYoung",
                "HouseTwenty",
                "InsectEPGRegularTrain",
                "InsectEPGSmallTrain",
                "MelbournePedestrian",
                "MixedShapesRegularTrain",
                "MixedShapesSmallTrain",
                "PickupGestureWiimoteZ",
                "PigAirwayPressure",
                "PigArtPressure",
                "PigCVP",
                "PLAID",
                "PowerCons",
                "Rock",
                "SemgHandGenderCh2",
                "SemgHandMovementCh2",
                "SemgHandSubjectCh2",
                "ShakeGestureWiimoteZ",
                "SmoothSubspace",
                "UMD", ]


@app.route('/helloReact')
def helloReact():
    return 'rrr react.'


@app.route('/getDataConf', methods=['POST'])
def getDataConf():
    dataset_name = request.json.get("dataset_name")

    conf = {}
    if dataset_name not in UCR_DATASETS:
        conf["num_class"] = 7  # TODO: generate data conf for hk air dataset
    else:
        with open(os.path.join(
                "./datasets/confs", dataset_name + ".json"), "r") as f:
            conf = json.load(f)
    return conf


@app.route('/getSimilarity', methods=["POST"])
def getSimilarity():
    dataset_name = request.json.get('dataset_name')
    embedding_method = request.json.get('embedding_method')
    marks_key = embedding_method + "_marks"

    res = {}

    with open("./data/sorted-similarity-matrix/" + str(dataset_name) + ".json", 'r') as f:
        data = json.load(f)

    res["data"] = data[embedding_method]
    res["mark"] = data[marks_key]

    return res


@app.route('/getCluster', methods=["POST"])
def getCluster():
    method = request.json.get('cluster_method')
    n_clusters = request.json.get('n_clusters')
    embedding_method = request.json.get('embedding_method')
    dataset_name = request.json.get("dataset_name")
    dim_reduction = request.json.get("dim_reduction")

    embedding_path = get_embedding_path_by_dataset(
        embedding_method, dataset_name)

    xs, ys, labels_cluster, labels = cluster_embedding(
        method, dataset_name, embedding_path, dim_reduction, n_clusters)
    res = {}
    res["data"] = np.array([xs, ys, labels, labels_cluster]).T.tolist()
    res["origin_label"] = labels
    res["cluster_label"] = labels_cluster.tolist()

    return res


@app.route('/getGroupLines_TH', methods=["POST"])
def getGroupLines_TH():
    mode = request.json.get('mode')  # 数据请求模式
    dataset_name = request.json.get("dataset_name")
    res = {}
    if dataset_name != '':
        if mode == 'pollutant':
            data, labels = load_HKAIR_ALL(mode)
            res["data"] = data
            res["label"] = labels
    return res


@app.route('/getGroupLinesSensor', methods=['POST'])
def getGroupLinesSensor():
    res = {}
    data = np.load(
        "./datasets/HK-RSP-NPY/HK_D.npy").reshape([25, 10, 305])

    res['data'] = data.tolist()
    res['times'] = TIMESTAMPS
    res['label'] = POLLUTANTS

    return res


@app.route('/getGroupLinesCluster', methods=['POST'])
def getGroupLinesCluster():
    cluster_label = request.json.get('cluster_label')
    group_line_list = request.json.get('group_line_list')
    res = {}

    cluster_label = [int(cl) for cl in cluster_label]

    k = max(cluster_label)  # num of cluster

    cluster_label = np.array(cluster_label).reshape([10, 5]).T
    data = np.load(
        "./datasets/HK-RSP-NPY/HK_D.npy").reshape([25, 10, 5, 61])
    data = np.transpose(data, (0, 2, 1, 3))
    
    idxes = [POLLUTANTS.index(item) for item in group_line_list]

    data = data[idxes]  # [3, 5, 10, 61] [污染物, 5年, 10个监测站, 61个时间戳]

    # 根据cluster label聚合数据
    data_res = []
    for pi in range(len(group_line_list)):
        data_in_pollus = []
        for year in range(5):
            data_in_year = []
            for label in range(int(k)+1):
                data_in_label = []
                a = data[pi, year, cluster_label[year]==label, :]
                if a.size!=0:
                    data_in_label.append(np.mean(np.array(a), axis=0).tolist())
                else:
                    data_in_label.append(np.zeros(61).tolist())
                data_in_year.append(data_in_label)
            data_in_pollus.append(data_in_year)
        data_res.append(data_in_pollus)
    data_res = np.array(data_res).reshape(len(idxes), 5, k+1, 61)
    
    maxes = [ np.max(data_res[i]) for i in range(data_res.shape[0]) ]

    res["max"] = maxes
    res["data"] = data_res.tolist()
    res['times'] = np.array(TIMESTAMPS).reshape([5, 61]).tolist()
    return res

@app.route('/getAvgLine', methods=["POST"])
def getAvgLine():
    mode = request.json.get('mode')  # 数据请求模式
    dataset_name = request.json.get("dataset_name")
    is_avg = request.json.get("is_avg")
    cluster_method = request.json.get('cluster_method')
    embed_method = request.json.get('embed_method')
    n_clusters = request.json.get('n_clusters')
    res = {}
    if dataset_name != '':
        if mode == 'UCR':
            data, labels, _, _ = load_UCR(dataset_name)
            n = data.shape[0]
            t = data.shape[1]
            if is_avg:
                # 1. 读取聚类标签
                # TODO: problems when cluster -> label -> avg line
                # embedding_path = get_embedding_path_by_dataset(
                #     embed_method, dataset_name)
                # _, _, labels, _ = cluster_embedding(
                #     cluster_method, dataset_name, embedding_path, n_clusters)
                # 2. 拼接标签与数据
                labels_2d = labels.reshape(n, 1)
                data_with_label = np.hstack((labels_2d, data))
                # 3. 根据聚类标签分组, 取平均
                columns = ['label']
                columns.extend([x for x in range(t)])
                data_with_label_df = pd.DataFrame(
                    data_with_label, columns=columns)
                mean_df = data_with_label_df.groupby(
                    data_with_label_df['label']).mean()
                group_labels = mean_df.index.tolist()

                res["data"] = mean_df.values.tolist()
                res["label"] = group_labels
            else:
                res["data"] = data.tolist()
                res["label"] = labels.tolist()
        elif mode == 'pollutant':
            data, labels = load_HKAIR_50(mode, dataset_name)
            if is_avg:
                mean_data = np.mean(data.reshape([10, 5, 61]), axis=1)
                res["data"] = mean_data.tolist()
                res["label"] = np.unique(np.array(labels)).tolist()
            else:
                res["data"] = data.tolist()
                res["label"] = labels
    return res


@app.route('/getRelationRect', methods=["POST"])
def getRelationRect():
    res = {}
    dataset_name = request.json.get("dataset_name")
    dim_p = request.json.get("dim_p")
    embedding_path = request.json.get("embedding_path")

    data, _, _, _ = load_UCR(dataset_name)
    W, H = nmf_transform(data, dim_p)  # N * P, P * T (300, 16) (16, 60)

    # embeddings = np.load(embedding_path)  # N * P (300, 16)

    # r = np.corrcoef(embeddings.T, W.T)
    # print(W.shape, H.shape, embeddings.shape, r.shape)

    dim_feature = H.shape[0]
    dim_series = H.shape[1]
    H = H.flatten()
    res["data"] = H.tolist()
    res["dim_feature"] = dim_feature
    res["dim_series"] = dim_series
    return res


@app.route('/getDenseLine', methods=["POST"])
def getDenseLine():
    mode = request.json.get("mode")
    dataset_name = request.json.get("dataset_name")

    if mode == 'UCR':
        dense_file = os.path.join(
            './data/dense-statistic', dataset_name + '.json')
    elif mode == 'pollutant':
        dense_file = os.path.join(
            './data/dense-statistic-hkair/Al.json')
    with open(dense_file, 'r') as f:
        res = json.load(f)

    return res


@app.route('/getDensePoint', methods=["POST"])
def getDensePoints():
    mode = request.json.get("mode")

    dataset_name = request.json.get("dataset_name")
    ts = int(request.json.get('time_stamp'))
    y = int(request.json.get('y'))
    if mode == 'UCR':
        data, _, _, _ = load_UCR(dataset_name)
    elif mode == 'pollutant':
        data, _ = load_HKAIR_50('pollutant', dataset_name)

    data_ten = np.array(data * 10, dtype=float)
    data_ten = np.round(data_ten, decimals=0)
    data_ten_t = data_ten.T

    candidates_index = [i for i, d in enumerate(
        data_ten_t[int(ts)]) if int(d) == y]

    candidates = []
    for c in candidates_index:
        candidates.append(data_ten[c])
        # counts[str(c)] = dict(collections.Counter(data_ten[c]))
    candidates_t = np.array(candidates).T

    res = {}
    for time_stamp in range(candidates_t.shape[0]):
        res_ts = list(collections.Counter(
            candidates_t[time_stamp]).keys())

        res[str(time_stamp)] = list(map(int, res_ts))
    return res


@app.route('/getHKGeoJson', methods=['POST'])
def getHKGeoJson():
    time_stamp = request.json.get("time_stamp")  # 时间戳
    range_bar_list = request.json.get("range_bar_list")
    pollu_idx = request.json.get("pollu_idx")

    time_stamp = TIMESTAMPS.index(time_stamp)

    res = {}
    # geo json
    geo_file = './data/hk-geo.json'
    with open(geo_file, 'r', encoding='utf-8') as f:
        res['hk'] = json.load(f)
    # sensor file
    sensor_file = './datasets/sensors.csv'
    sensors_df = pd.read_csv(sensor_file, sep=",", header=0)
    res['sensor'] = np.array(sensors_df).tolist()

    # data
    data = np.load(
        './datasets/HK-RSP-NPY/HK_D.npy').reshape([25, 10, 305])
    data_at = data[pollu_idx, :, time_stamp]
    res['data'] = data_at.tolist()

    # punch line data
    idxes = [POLLUTANTS.index(item) for item in range_bar_list]

    punch_data = np.load(
        "./datasets/HK-RSP-NPY/HK_D.npy").reshape([25, 10, 305])
    punch_data = punch_data[idxes, :, time_stamp]

    punch_max = np.amax(punch_data.T, axis=0) 

    res['punch'] = punch_data.T.tolist()
    res['punch_max'] = punch_max.tolist()

    return res


@app.route('/getClusterCompare', methods=['POST'])
def getClusterCompare():
    dataset_name = request.json.get("dataset_name")
    changed_labels = request.json.get("changed_labels")
    caseIndex = 3
    data = np.load("./datasets/HK-RSP-EMBEDDING-NPY/" + dataset_name+".npy")
    data = data.reshape([50, -1])

    # data = np.load("./datasets/HK-RSP-NPY/HK_D.npy")  # [25, 50, 61]
    # idx = POLLUTANTS.index(dataset_name)
    # data = data[idx]

    res_list = multi_process_cluster(data, max_thread=6, max_cluster=8)
    nums = []
    center_dists = []
    inter_dists = []
    SC = []
    CH = []
    DB = []
    
    for item in res_list:
        nums.append(item[0])
        center_dists.append(item[1])
        inter_dists.append(item[2])
        SC.append(item[3])
        CH.append(item[4])
        DB.append(item[5])

    res = {}

    # 手动修改标签
    if len(changed_labels) != 0:
        y_manual = np.array([int(l) for l in changed_labels])
        y_unique = np.unique(y_manual)
        print("y_manual",y_manual)
        print("y_unique",y_unique)
        manual_member = []
        manual_center_dist = []
        manual_inter_dist = []

        # centroids = []
        for yl in y_unique:
            idxes = np.where(y_manual==yl)  # label 为 yl 的样本索引
            cluster_data = data[idxes]  # 根据索引取label 为 yl 的样本
            # print("cluster_data",cluster_data)
            ctr, ctr_idx = select_center(cluster_data)
            manual_member.append(cluster_data.shape[0])
            manual_center_dist.append(cdist(cluster_data, [ctr]).mean())

            # density
            manual_dis_matrix = np.triu(np.sqrt(np.sum((cluster_data[:, np.newaxis, :] - cluster_data[np.newaxis, :, :]) ** 2, axis=2)), k=1)
            manual_inter_dist.append(float(np.mean(manual_dis_matrix)))
        # SC = np.append(SC, silhouette_score(data, y_manual))
        # CH = np.append(CH, calinski_harabasz_score(data, y_manual))
        # DB = np.append(DB, davies_bouldin_score(data, y_manual))
        SC[caseIndex]= (silhouette_score(data, y_manual))
        CH[caseIndex]= (calinski_harabasz_score(data, y_manual))
        DB[caseIndex] = (davies_bouldin_score(data, y_manual))
        # nums.append(manual_member)
        nums[caseIndex]=manual_member
        # center_dists.append(manual_center_dist)
        # inter_dists.append(manual_inter_dist)
        center_dists[caseIndex]=manual_center_dist
        inter_dists[caseIndex]=manual_inter_dist

    SC_max = float(max(SC))
    CH_max = float(max(CH))
    DB_max = float(max(DB))

    radar_indicators = np.array([SC, CH, DB]).T

    res['radar'] = radar_indicators.tolist()
    res['maxes'] = [SC_max, CH_max, DB_max]
    res['name'] = ['SC', 'CH', 'DB']
    res['size'] = nums
    res['variance'] = center_dists
    res['density'] = inter_dists
    # print("ressssssssss",res)
    return res


@app.route('/getRangeBar', methods=['POST'])
def getRangeBar():
    time_stamp = request.json.get('time_stamp')
    pollutant = request.json.get('pollutant')
    # range_bar_list = request.json.get('range_bar_list')

    # time_stamp = int(time_stamp % 61)
    time_stamp = TIMESTAMPS.index(time_stamp) % 61

    res = {}

    data = np.load('./datasets/HK-RSP-NPY/HK_T.npy')  # 50 * 61 * 25
    data_ts = data.swapaxes(0, 1)  # 61 * 50 * 25
    data_ts = data.swapaxes(1, 2)  # 61 * 25 * 50

    # idxes = [range_bar_list.index(item) for item in range_bar_list]

    data_at_ts = data_ts[int(time_stamp)]  # 取当前时间点
    data_avg = data_at_ts.mean(axis=1).tolist()
    data_max = data_at_ts.max(axis=1).tolist()
    data_min = data_at_ts.min(axis=1).tolist()

    # 对确定的污染物 每个时刻在不同城市的min max avg
    data_ts = data_ts.swapaxes(0, 1).reshape([25, 305, 10])  # 25 * 61 * 50
    data_ts_at_ts = data_ts[int(pollutant)]  # 取当前污染物
    data_ts_avg = data_ts_at_ts.mean(axis=1).tolist()
    data_ts_max = data_ts_at_ts.max(axis=1).tolist()
    data_ts_min = data_ts_at_ts.min(axis=1).tolist()

    res_data_at_time = np.array([data_min, data_max, data_avg]).T.tolist()
    res_data_at_pollu = np.array(
        [data_ts_min, data_ts_max, data_ts_avg]).T.tolist()
    res["data"] = res_data_at_time
    res["dataP"] = res_data_at_pollu
    res["pollutant"] = POLLUTANTS
    res["times"] = TIMESTAMPS
    return res


@app.route('/getFeatureContribution', methods=['POST'])
def getFeatureContribution():
    dataset_name = request.json.get('dataset_name')
    time_stamp = request.json.get('time_stamp')
    label_set = request.json.get('label_set')
    # is_self = request.json.get('is_self')
    res = {}

    label_set = [int(x) for x in label_set]
    k_cluster = max(label_set)

    cluster_label_path = "./datasets/ccpca-label.npy"
    np.save(cluster_label_path, np.array(label_set))

    # 计算 Feature Contribution 和 Transition
    idx_p = POLLUTANTS.index(dataset_name)
    # idx_t = TIMESTAMPS.index(time_stamp)
    command = "python3 ./utils/ccpca/ccpca/sample.py {} {} {}".format(idx_p, time_stamp, cluster_label_path)
    p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdoutput = p.communicate()[0].decode('utf-8')
    stdoutput = stdoutput.replace('\n', '')

    std_res = ast.literal_eval(stdoutput)

    ft = std_res[0:k_cluster+1]
    fc = std_res[k_cluster+1:]
    res["fc"] = fc
    res["ft"] = ft
    
    return res

@app.route('/getBoxPlot', methods=['POST'])
def getBoxPlot():
    time_stamp = request.json.get('time_stamp')
    pollutant = request.json.get('pollutant')

    time_stamp = TIMESTAMPS.index(time_stamp)

    res ={}

    # 对某一个特定的污染物
    data = np.load('./datasets/HK-RSP-NPY/HK_D.npy').reshape([25, 305, 10])  # [25, 50, 61] -> [25, 305, 10]
    data_at_pollu = data[int(pollutant)]

    # 对某一个确定的时间点
    data = np.load('./datasets/HK-RSP-NPY/HK_T.npy').swapaxes(0, 2)  # [50, 61, 25] -> [25, 61, 50]
    data = data.reshape([25, 305, 10])   # [25, 61, 50] => [25, 305, 10]
    data = data.swapaxes(0, 1)  # [25, 305, 10] => [305, 25, 10]
    data_at_time = data[time_stamp]

    res["dataT"] = data_at_time.tolist()
    res["dataP"] = data_at_pollu.tolist()
    return res

@app.route('/getContrastHeat', methods=['POST'])
def getContrastHeat():
    time_stamp = request.json.get('time_stamp')
    time_range = request.json.get('time_range')
    res = {}
    time_stamp = TIMESTAMPS.index(time_stamp) 
    data = np.load(
        "./datasets/HK-RSP-NPY/HK_D.npy").reshape([25, 10, 305])
    data = data.swapaxes(0, 1).swapaxes(1, 2).swapaxes(0, 1) # 305 * 10 * 25
    print(type(data))
    if time_stamp-time_range>=0 and time_stamp+time_range<=360:
        res["data"] = data.tolist()[time_stamp-time_range:time_stamp+time_range+1]
    elif time_stamp-time_range<0:
        start = 0
        end = time_stamp+time_range
        res["data"] = data.tolist()[start:end+1]
    elif time_stamp+time_range>360:
        start = time_stamp-time_range
        end = 360
        res["data"] = data.tolist()[start:end+1] # times*10*25
    for i in range(len(res["data"])):
        #list转变为np.array
        res["data"][i] = np.array(res["data"][i])
        res["data"][i]=res["data"][i].swapaxes(0,1).tolist()
    res["data"] = np.array(res["data"])
    res["data"] = res["data"].swapaxes(0,1).tolist()
    res['times'] = TIMESTAMPS[time_stamp-time_range:time_stamp+time_range+1]   # 305 
    res['label'] = POLLUTANTS   # 25
    return res

if __name__ == '__main__':
    # app.debug = True   # 开启调试模式, 代码修改后服务器自动重新载入，无需手动重启
    app.run()
