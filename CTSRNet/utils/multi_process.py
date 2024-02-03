import queue
import threading
import time
import numpy as np
import pandas as pd
import os
import ast

from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import subprocess

from utils.constants import POLLUTANTS

path = os.getcwd()  # 获取当前路径


class dimReductionThread(threading.Thread):
    def __init__(self, func, args, name=''):
        threading.Thread.__init__(self)
        self.name = name
        self.func = func
        self.args = args
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result
        except Exception:
            return None
        

def _cluster(n_clusters, data):
    num_instance = data.shape[0]
    dim_instance = data.shape[1]

    # 自动算法聚类
    k_means = KMeans(n_clusters=n_clusters)
    y = k_means.fit_predict(data)
    # centroid
    centroids = k_means.cluster_centers_

    y_T = np.array(y).reshape([num_instance, 1]).tolist()
    y_set = np.unique(y)

    columns = ['label']
    for i in range(dim_instance):
        columns.append(i)

    clusters_all = np.append(y_T, data, axis=1)
    clusters_all_df = pd.DataFrame(data=clusters_all, columns=columns)
    cluster_groups = clusters_all_df.groupby(['label'])

    avg_center_dist = []  # 每个簇内的样本到中心点之间的距离的均值
    avg_inter_dist = []  # density 每个簇内所有数据点之间的平均距离
    cluster_num = []  # 每个簇内的样本数量

    for yi, ys in enumerate(y_set):
        center = np.array(centroids[yi]).reshape([1, -1])
        cluster = np.array(
            list(cluster_groups.get_group(ys).values))[:, :-1]
        cluster_num.append(cluster.shape[0])

        avg_center_dist.append(cdist(cluster, center).mean())

        dis_inter_matrix = np.triu(np.sqrt(np.sum((cluster[:, np.newaxis, :] - cluster[np.newaxis, :, :]) ** 2, axis=2)), k=1)
        avg_inter_dist.append(np.mean(dis_inter_matrix))

    return cluster_num, avg_center_dist, avg_inter_dist, silhouette_score(data, y), calinski_harabasz_score(data, y), davies_bouldin_score(data, y)


def _get_ft(pi, cluster_label_path):
    command = "python3 ts2vec/utils/ccpca/ccpca/sample.py {} {} {}".format(pi, cluster_label_path, "1")
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdoutput = process.communicate()[0].decode('utf-8')
    stdoutput = stdoutput.replace('\n', '')
    ft = ast.literal_eval(stdoutput)
    return ft
    # fts.append(ast.literal_eval(stdoutput))

def multi_process_ft(cluster_label_path, max_thread):
    results = []

    print("threading start")
    start = time.time()
    # 创建队列,队列大小限制线程个数
    q = queue.Queue(maxsize=max_thread)
    # 多线程降维
    # TODO: 修改POLLUTANTS取值范围
    for pi, p in enumerate(POLLUTANTS[:3]):
        t = dimReductionThread(func=_get_ft, args=(pi, cluster_label_path), name='feature_trans_{}'.format(pi))
        q.put(t)
        # 队列队满
        if q.qsize() == max_thread:
            # 记录线程
            # 从队列中取线程 直至队列为空
            joinThread = []
            while q.empty() != True:
                t = q.get()
                joinThread.append(t)
                t.start()
            # 终止所有线程
            for t in joinThread:
                t.join()
                results.append(t.get_result())
    # 清空剩余线程
    restThread = []
    while q.empty() != True:
        t = q.get()
        restThread.append(t)
        t.start()
    for t in restThread:
        t.join()
        results.append(t.get_result())
    end = time.time() - start
    print("similarity thread cost: "+str(end))

    return results



def multi_process_cluster(data, max_thread, max_cluster):
    results = []

    print("threading start")
    start = time.time()
    # 创建队列,队列大小限制线程个数
    q = queue.Queue(maxsize=max_thread)
    # 多线程降维
    for nc in range(2, max_cluster):
        t = dimReductionThread(func=_cluster, args=(nc, data), name='many_cluster_{}'.format(id))
        q.put(t)
        # 队列队满
        if q.qsize() == max_thread:
            # 记录线程
            # 从队列中取线程 直至队列为空
            joinThread = []
            while q.empty() != True:
                t = q.get()
                joinThread.append(t)
                t.start()
            # 终止所有线程
            for t in joinThread:
                t.join()
                results.append(t.get_result())
    # 清空剩余线程
    restThread = []
    while q.empty() != True:
        t = q.get()
        restThread.append(t)
        t.start()
    for t in restThread:
        t.join()
        results.append(t.get_result())
    end = time.time() - start
    print("similarity thread cost: "+str(end))

    return results
