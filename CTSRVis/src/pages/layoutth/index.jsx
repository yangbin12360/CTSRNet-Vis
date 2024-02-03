import { Row, Col } from "antd";
import { useEffect, useState } from "react";

import ControlBar from "../../components/ControlBar";
import ClusterScatter from "../../components/ClusterScatter";
import Map from "../../components/Map";
import ClusterPage from "../../components/ClusterPage";
// import RangeBar from "../../components/RangeBar";
import Box from "../../components/Box/box";
import GroupLines from "../../components/GroupLines";
import FeatureContribution from "../../components/FeactureContribution";
import BoxPlot from "../../components/BoxPlot";
import FeatureTransition from "../../components/FeatureTransition";
import ContrastHeat from "../../components/ContrastHeat";

import "./index.less";
const LayoutTh = () => {
  // ----------------------- states --------------------------
  const [dataset, setDataset] = useState("As");
  const [feature, setFeature] = useState("tsrnet");
  const [cluster, setCluster] = useState("kmeans");
  const [numOfCluster, setNumOfCluster] = useState(4);
  const [focusIndex, setFocusIndex] = useState(numOfCluster);
  // const [numOfManualCluster, setNumOfManualCluster] = useState(8)  // 手动修改的Cluster结果的index
  const [lineGroupType, setLineGroupType] = useState("sensor");
  const [dimReduction, setDimReduction] = useState("tsne");
  // prettier-ignore
  const [rangeBarList, setRangeBarList] = useState(["As","Be","Cd","Ni","Hg",]);
  const [groupLineList, setGroupLineList] = useState(["As", "Be", "Cd","Ni","Hg"]);
  const [rangeBarTimeStampIdx, setRangeBarTimeStampIdx] = useState("20171205"); // 在GroupLines中选择时间点，更新Timestamp Bar
  const [rangeBarPollutantIdx, setRangeBarPollutantIdx] = useState(0);
  const [fcData, setFcData] = useState(null); // FC 视图数据
  const [clusterLabel, setClusterLabel] = useState([]); // 聚类返回的标签, 用于分组画GroupLines
  const [changedLabel, setChangedLabel] = useState([]); // 手动修改后的 label
  const [taskState, setTaskState] = useState("anomaly");

  return (
    <div style={{ width: "100vw", height: "100vh", overflow: "hidden" }}>
      <Row style={{ width: "100%", height: "100%" }}>
        <Col span={16} id="left" style={{ width: "100%", height: "100%" }}>
          <div style={{ height: "65%" }} id="left-top">
            <Row style={{ width: "100%", height: "100%" }}>
              <Col span={8} style={{ height: "100%" }}>
                <div style={{ height: "40%" }}>
                  <Box
                    title={"Control Bar"}
                    component={
                      <ControlBar
                        dataset={dataset}
                        feature={feature}
                        handleFeatureChange={(c) => setFeature(c)}
                        cluster={cluster}
                        dimReduction={dimReduction}
                        lineGroupType={lineGroupType}
                        rangeBarList={rangeBarList}
                        groupLineList={groupLineList}
                        rangeBarTimeStampIdx={rangeBarTimeStampIdx}
                        taskType={taskState}
                        handleModeChange={
                          (c) => {
                            setDataset(c)
                          }
                        }
                        handleClusterChange={(c) => setCluster(c)}
                        handleDimReductionChange={(e) =>
                          setDimReduction(e.target.value)
                        }
                        handleLineGroupTypeChange={(e) =>
                          setLineGroupType(e.target.value)
                        }
                        handleRangeBarListChange={(c) => {
                          // console.log("rangeBarList,",rangeBarList);
                          setRangeBarList(c)
                        }}
                        handleGroupLineListChange={(c) =>{ 
                          setRangeBarList(c)
                          setGroupLineList(c)
                        }}
                        handleTaskTypeChange={(c) => setTaskState(c)}
                      />
                    }
                  />
                </div>
                <div style={{ height: "60%" }}>
                  <Box
                    title={"Geographic Information View"}
                    component={
                      <Map
                        rangeBarTimeStampIdx={rangeBarTimeStampIdx}
                        rangeBarList={rangeBarList}
                        rangeBarPollutantIdx={rangeBarPollutantIdx}
                      />
                    }
                  />
                </div>
              </Col>
              <Col span={16} style={{ height: "100%" }}>
                {/* <div style={{ height: "40%" }}>
                  <Box
                    title={"D / T Boxplot"}
                    component={
                      <BoxPlot
                        rangeBarList={rangeBarList}
                        rangeBarTimeStampIdx={rangeBarTimeStampIdx}
                        rangeBarPollutantIdx={rangeBarPollutantIdx}
                      />
                    }
                  />
                </div> */}
                <div style={{ height: "100%" }}>
                  <Box
                    title={"Group Lines"}
                    component={
                      <GroupLines
                        lineGroupType={lineGroupType}
                        clusterLabel={clusterLabel}
                        groupLineList={groupLineList}
                        setRangeBarTimeStampIdx={setRangeBarTimeStampIdx}
                        setRangeBarPollutantIdx={setRangeBarPollutantIdx}
                      />
                    }
                  />
                </div>
              </Col>
            </Row>
          </div>
          <div style={{ height: "35%" }} id="left-bottom">
            <Row style={{ width: "100%", height: "100%" }}>
              <Col span={8} style={{ height: "100%" }}>
                <Box
                  title={"Cluster Scatter"}
                  component={
                    <ClusterScatter
                      dataset={dataset}
                      cluster={cluster}
                      numOfCluster={numOfCluster}
                      embeddingMethod={feature}
                      dimReduction={dimReduction}
                      setClusterLabel={setClusterLabel}
                      setChangedLabel={setChangedLabel}
                    />
                  }
                />
              </Col>
              <Col span={16} style={{ height: "100%" }}>
                <Box
                  title={"Clustering Comparison"}
                  component={
                    <ClusterPage
                      dataset={dataset}
                      cluster={cluster}
                      focusIndex={focusIndex}
                      changedLabel={changedLabel}
                      handleNumOfClusterChange={setNumOfCluster}
                      handleFocusIndex={setFocusIndex}
                    />
                  }
                />
              </Col>
            </Row>
          </div>
        </Col>
          <Col span={8} id="right">
            {taskState === "anomaly"?
            <Box
              title={"ContrastHeat"}
              component={
                <ContrastHeat
                timeStamp={rangeBarTimeStampIdx}
                />
              }
            >
            </Box>:<>
              <div style={{ height: "26%" }}>
              <Box
                title={"Feature Transition"}
                component={
                  <FeatureTransition
                    dataset={dataset}
                    clusterLabel={clusterLabel}
                    setFcData={setFcData}
                  />
                }
              />
            </div>
            <div style={{ height: "74%" }}>
              <Box
                title={"Feature Contribution"}
                component={<FeatureContribution fcData={fcData} />}
              />
            </div>
            </>}
        </Col>
      </Row>
    </div>
  );
};

export default LayoutTh;
