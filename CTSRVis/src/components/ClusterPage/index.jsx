import "./index.less";
import { getClusterCompare } from "../../apis/api";
import ClusterRadar from "../ClusterRadar";
import ClusterBar from "../ClusterBar";
import ClusterDetail from "../ClusterDetail";
import { useEffect, useRef, useState } from "react";

const ClusterPage = (props) => {
  const {
    dataset,
    cluster,
    focusIndex,
    changedLabel,
    handleNumOfClusterChange,
    handleFocusIndex,
  } = props;
  const [radar, setRadar] = useState([]);
  const [maxes, setMaxes] = useState([]);
  const [name, setName] = useState([]);
  const [dists, setDists] = useState([]);
  const [nums, setNums] = useState([]);
  const [detail, setDetail] = useState({});

  const cluPageRef = useRef(null);

  const handleClusterBarClick = (e) => {
    let i = e.target.parentElement.parentElement
      .getAttribute("id")
      .split("-")[3];
    i = parseInt(i) + 1;

    if (i < 7) {
      // TODO: 最大聚类数
      handleNumOfClusterChange(i);
      handleFocusIndex(i);
    } else {
      handleFocusIndex(i);
    }
  };

  useEffect(() => {
    // isFcDataChange == true时, 根据新的Label, 计算相关指标, 添加到现有绘图数据中
    getClusterCompare(dataset, cluster, changedLabel).then((res) => {
      console.log("dataset, cluster, changedLabel",dataset);
      console.log("cluster",cluster);
      console.log("changeLabel",changedLabel);
      const { radar, maxes, name, variance, size, density } = res;
      setRadar(radar);
      setMaxes(maxes);
      setName(name);
      setDists(variance);
      setNums(size);
      setDetail({ size, variance, density });
    });
  }, [dataset, cluster, changedLabel]);

  return (
    <div className="cluster-page" ref={cluPageRef}>
      <div className="radar-list">
        {dists.map((item, index) => (
          <div key={index} className="radar-list-item">
            <ClusterRadar series={radar[index]} name={name} maxes={maxes} />
          </div>
        ))}
      </div>
      <div className="bar-list">
        {nums.map((item, index) => (
          <div
            key={index}
            className="bar-list-item"
            onClick={handleClusterBarClick}
          >
            <ClusterBar series={item} index={index} />
          </div>
        ))}
      </div>
      <div className="detail-cluster">
        {Object.keys(detail).map((item, index) => (
          <ClusterDetail
            key={index}
            detail={detail[item]}
            title={item}
            cIndex={focusIndex}
          />
        ))}
      </div>
    </div>
  );
};

export default ClusterPage;
