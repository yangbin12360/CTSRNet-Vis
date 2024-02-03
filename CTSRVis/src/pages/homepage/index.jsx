import { useState, useEffect } from "react";

import "./index.less";

import TitleBar from "../../components/TitleBar";
import ControlBar from "../../components/ControlBar";
import ClusterScatter from "../../components/ClusterScatter";
import SimilarityMatrix from "../../components/SimilarityMatrix";
import AvgLine from "../../components/AvgLine";

import { getDataConf } from "../../apis/api";

const HomePage = () => {
  const [dataset, setDataset] = useState("SyntheticControl");
  const [feature, setFeature] = useState("seanet");
  const [cluster, setCluster] = useState("kmeans");
  const [numOfCluster, setNumOfCluster] = useState(6);
  const [maxNumOfCluster, setMaxNumOfCluster] = useState(0);
  const [isTsne, setIsTsne] = useState(true);
  const [isAvg, setIsAvg] = useState(true); // 折线图是否展示簇平均折线

  useEffect(() => {
    getDataConf(dataset).then((res) => {
      let max = res["num_class"];
      if (numOfCluster > max) {
        setNumOfCluster(max);
      }
      setMaxNumOfCluster(max);
    });
  }, [dataset, numOfCluster]);

  return (
    <div className="home-page">
      <div className="row-one">
        <div className="control-wrap">
          <TitleBar title="Control Bar" />
          <ControlBar
            dataset={dataset}
            feature={feature}
            handleFeatureChange={(c) => setFeature(c)}
            cluster={cluster}
            numOfCluster={numOfCluster}
            maxNumOfCluster={maxNumOfCluster}
            handleDatasetChange={(c) => setDataset(c)}
            handleClusterChange={(c) => setCluster(c)}
            handleNumOfClusterChange={(c) => setNumOfCluster(c)}
            handleTsneSwitchChange={(checked) => setIsTsne(checked)}
            handleLineSwitchChange={(checked) => setIsAvg(checked)}
          />
        </div>

        <div className="similarity-wrap">
          <TitleBar title="Similarity Matrix" />
          <SimilarityMatrix dataset={dataset} embeddingMethod={feature} />
        </div>
        <div className="cluster-wrap">
          <TitleBar title="Cluster Scatter" />
          <ClusterScatter
            dataset={dataset}
            cluster={cluster}
            numOfCluster={numOfCluster}
            embeddingMethod={feature}
            isTsne={isTsne}
          />
        </div>

        <div className="avgline-wrap">
          <TitleBar title="Average Series" />
          <AvgLine
            dataset={dataset}
            isAvg={isAvg}
            feature={feature}
            cluster={cluster}
            numOfCluster={numOfCluster}
          />
        </div>
      </div>

      <div className="row-two">row two</div>
    </div>
  );
};

export default HomePage;
