import { get, post } from "./http";

export const helloReact = () => {
  return get("/helloReact");
};

// 获取数据集基本信息, 用于更新控制台中对应属性默认值
export const getDataConf = (dataset_name) => {
  return post("getDataConf", {
    dataset_name: dataset_name,
  });
};

// 获取相似度计算结果,用于绘制相似度矩阵
export const getSimilarity = (dataset_name, embedding_method) => {
  return post("getSimilarity", {
    dataset_name: dataset_name,
    embedding_method: embedding_method,
  });
};

// 获取聚类结果, 用于绘制散点图
export const getCluster = (
  method,
  n_clusters,
  embedding_method,
  dataset_name,
  dim_reduction
) => {
  return post("getCluster", {
    cluster_method: method,
    n_clusters: n_clusters,
    embedding_method: embedding_method,
    dataset_name: dataset_name,
    dim_reduction: dim_reduction,
  });
};

// 获取时序数据一定范围内的平均变化, 用于绘制平均折线图
export const getAvgLine = (
  mode,
  dataset_name,
  is_avg,
  embed_method,
  cluster_method,
  n_clusters
) => {
  return post("getAvgLine", {
    mode: mode,
    dataset_name: dataset_name,
    is_avg: is_avg,
    embed_method: embed_method,
    cluster_method: cluster_method,
    n_clusters: n_clusters,
  });
};

export const getGroupLinesSensor = () => {
  return post("getGroupLinesSensor", {
    // mode: mode,
  });
};

/**
 * 
 * @param {聚类标签} cluster_label 
 * @param {指标列表} group_line_list 
 * @returns 
 */
export const getGroupLinesCluster = (cluster_label, group_line_list) => {
  return post("getGroupLinesCluster", {
    cluster_label: cluster_label,
    group_line_list: group_line_list
  })
}

// 获取特征的相关性结果，用于绘制矩阵图
export const getRelationRect = (dataset_name, dim_p, embedding_path) => {
  return post("getRelationRect", {
    dataset_name: dataset_name,
    dim_p: dim_p,
    embedding_path: embedding_path,
  });
};

// 获取时序密度图数据
export const getDenseLine = (mode, dataset_name) => {
  return post("getDenseLine", {
    mode: mode,
    dataset_name: dataset_name,
  });
};

// 密度图交互, 根据hover的数据点，返回关联点
export const getDensePoint = (mode, dataset_name, time_stamp, y) => {
  return post("getDensePoint", {
    mode: mode,
    dataset_name: dataset_name,
    time_stamp: time_stamp,
    y: y,
  });
};

// 获取香港行政区划地理文件
export const getHKGeoJson = (pollu_idx, time_stamp, range_bar_list) => {
  return post("getHKGeoJson", {
    pollu_idx: pollu_idx,
    time_stamp: time_stamp,
    range_bar_list: range_bar_list,
  });
};

// 获取聚类对比的聚合数据
export const getClusterCompare = (dataset_name, cluster_method, changed_labels) => {
  return post("getClusterCompare", {
    dataset_name: dataset_name,
    cluster_method: cluster_method,
    changed_labels: changed_labels
    // is_manual: is_manual
  });
};

// 获取RangeBar数组
export const getRangeBar = (time_stamp, pollutant) => {
  return post("getRangeBar", {
    time_stamp: time_stamp,
    pollutant: pollutant,
    // range_bar_list: range_bar_list,
  });
};


/**
 * 
 * @param {当前污染物指标} dataset_name 
 * @param {选择时间点} time_stamp
 * @param {当前聚类标签} label_set 
 * @returns 
 */
export const getFeatureContribution = (dataset_name, time_stamp, label_set) => {
  return post("getFeatureContribution", {
    dataset_name: dataset_name,
    time_stamp: time_stamp,
    label_set: label_set,
  })
}

/**
 * 获取盒须图数据
 * @param {选择的时间点} time_stamp 
 * @param {选择的污染物list} pollutant 
 * @returns 
 */
export const getBoxPlot = (time_stamp, pollutant) => {
  return post("getBoxPlot", {
    time_stamp: time_stamp,
    pollutant: pollutant,
  })
}

/**
 * 获取对比热力图数据
 * @param {选择的时间点} time_stamp
 * @param {选择的时间范围} time_range 
 * @returns
 */
export const getContrastHeat = (time_stamp, time_range) => {
  return post("getContrastHeat", {
    time_stamp: time_stamp,
    time_range: time_range,
  })
}