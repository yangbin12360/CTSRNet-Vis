import { useState, useEffect } from "react";

import { Layout, Model } from "flexlayout-react";

import config from "../../utils/layoutconfig.json";

import DenseLine from "../../components/DenseLine";
import ControlBar from "../../components/ControlBar";
import ClusterScatter from "../../components/ClusterScatter";
import SimilarityMatrix from "../../components/SimilarityMatrix";
import TitleBar from "../../components/TitleBar";
import AvgLine from "../../components/AvgLine";
import Map from "../../components/Map";
import ClusterPage from "../../components/ClusterPage";
import RangeBar from "../../components/RangeBar";

import { getDataConf } from "../../apis/api";

import "flexlayout-react/style/underline.css";
import GroupLines from "../../components/GroupLines";
import LayoutTh from "../layoutth/index"

const AnalysisPage = () => {
  // ----------------------- states --------------------------
  const [mode, setMode] = useState("pollutant");
  const [dataset, setDataset] = useState("RSP");
  const [feature, setFeature] = useState("seanet");
  const [cluster, setCluster] = useState("kmeans");
  const [numOfCluster, setNumOfCluster] = useState(6);
  const [maxNumOfCluster, setMaxNumOfCluster] = useState(0);
  const [lineGroupType, setLineGroupType] = useState("sensor");
  const [isTsne, setIsTsne] = useState(true);
  const [isAvg, setIsAvg] = useState(true); // 折线图是否展示簇平均折线
  const [timeStamp, setTimeStamp] = useState(0); // 地图展示的时间切片的索引
  const [rangeBarList, setRangeBarList] = useState([
    "As",
    "Be",
    "Cd",
    "Ni",
    "Cr",
    "Hg",
  ]);
  const [groupLineList, setGroupLineList] = useState(["As", "Be", "Cd"]);
  const [rangeBarTimeStampIdx, setRangeBarTimeStampIdx] = useState(0); // 在GroupLines中选择时间点，更新Timestamp Bar
  const [rangeBarPollutantIdx, setRangeBarPollutantIdx] = useState(0);

  useEffect(() => {
    getDataConf(dataset).then((res) => {
      let max = res["num_class"];
      if (numOfCluster > max) {
        setNumOfCluster(max);
      }
      setMaxNumOfCluster(max);
    });
  }, [dataset, numOfCluster]);

  // const model = Model.fromJson(config);
  // const factory = (node) => {
  //   var component = node.getComponent();
  //   switch (component) {
  //     case "control-panel":
  //       return (
  //         <ControlBar
  //           feature={feature}
  //           handleFeatureChange={(c) => setFeature(c)}
  //           cluster={cluster}
  //           numOfCluster={numOfCluster}
  //           lineGroupType={lineGroupType}
  //           maxNumOfCluster={maxNumOfCluster}
  //           rangeBarList={rangeBarList}
  //           groupLineList={groupLineList}
  //           handleModeChange={(c) => {
  //             let [mode, dataset] = c;
  //             setMode(mode);
  //             setDataset(dataset);
  //           }}
  //           handleClusterChange={(c) => setCluster(c)}
  //           handleNumOfClusterChange={(c) => setNumOfCluster(c)}
  //           handleTsneSwitchChange={(checked) => setIsTsne(checked)}
  //           handleLineSwitchChange={(checked) => setIsAvg(checked)}
  //           handleLineGroupTypeChange={(c) => setLineGroupType(c)}
  //           handleRangeBarListChange={(c) => setRangeBarList(c)}
  //           handleGroupLineListChange={(c) => setGroupLineList(c)}
  //         />
  //       );
  //     case "dense-line":
  //       return (
  //         <RangeBar
  //           rangeBarList={rangeBarList}
  //           rangeBarTimeStampIdx={rangeBarTimeStampIdx}
  //           rangeBarPollutantIdx={rangeBarPollutantIdx}
  //         />
  //       );
  //     // return <DenseLine mode={mode} dataset_name={dataset} />;
  //     // return (
  //     //   <AvgLine
  //     //     mode={mode}
  //     //     dataset={dataset}
  //     //     isAvg={isAvg}
  //     //     feature={feature}
  //     //     cluster={cluster}
  //     //     numOfCluster={numOfCluster}
  //     //     onTimeStampChange={setTimeStamp}
  //     //   />
  //     // );
  //     case "cluster-scatter":
  //       return (
  //         <ClusterScatter
  //           mode={mode}
  //           dataset={dataset}
  //           cluster={cluster}
  //           numOfCluster={numOfCluster}
  //           embeddingMethod={feature}
  //           isTsne={isTsne}
  //         />
  //       );
  //     // case "similarity-matrix":
  //     //   return <SimilarityMatrix dataset={dataset} embeddingMethod={feature} />;
  //     case "title-bar":
  //       return <TitleBar title="title" />;
  //     case "avg-line":
  //       return (
  //         <GroupLines
  //           groupLineList={groupLineList}
  //           setRangeBarTimeStampIdx={setRangeBarTimeStampIdx}
  //           setRangeBarPollutantIdx={setRangeBarPollutantIdx}
  //         />
  //       );
  //     case "map":
  //       return <Map mode={mode} dataset={dataset} timeStamp={timeStamp} />;
  //     case "cluster-radar":
  //       // return <ClusterPage mode={mode} dataset={dataset} cluster={cluster} />;
  //       return <TitleBar title="title" />;

  //     default:
  //       return <></>;
  //   }
  // };

  return (
    <div style={{ width: "100vw", height: "100vh" }}>
      <LayoutTh></LayoutTh>
    </div>
  );
};

export default AnalysisPage;
