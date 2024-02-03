import { Select, Radio } from "antd";

import { POLLUTANTLIST_25 } from "../../utils/constants";
import "./index.less";

const ControlBar = (props) => {
  const {
    dataset,
    feature,
    handleFeatureChange,
    cluster,
    lineGroupType,
    taskType,
    dimReduction,
    rangeBarList,
    groupLineList,
    rangeBarTimeStampIdx,
    handleModeChange,
    handleClusterChange,
    handleDimReductionChange,
    handleLineGroupTypeChange,
    handleRangeBarListChange,
    handleGroupLineListChange,
    handleTaskTypeChange
  } = props;

  return (
    <div className="control">
      <div className="control-sensors">
        <div className="control-name">Dataset</div>
        <div className="control-sensors-select">
          <Select
            defaultValue={dataset}
            style={{ width: 90 }}
            size="small"
            onChange={handleModeChange}
            options={POLLUTANTLIST_25}
          />
        </div>
        <div className="control-sensors-text">{rangeBarTimeStampIdx}</div>
      </div>

      <div className="control-feature">
        <div className="control-name">Embed</div>
        <div className="control-feature-select">
          <Select
            defaultValue={feature}
            style={{ width: 90 }}
            size="small"
            onChange={handleFeatureChange}
            options={[
              {
                value: "paa",
                label: "PAA",
              },
              {
                value: "seanet",
                label: "SEANet",
              },
              {
                value: "ts2vec",
                label: "TS2Vec",
              },
              {
                value: "tsrnet",
                label: "CTSRNet",
              },
            ]}
          />
        </div>
      </div>

      <div className="control-cluster">
        <div className="control-name">Cluster</div>
        <div className="control-cluster-select">
          <div className="control-cluster-select-all">
            <div className="control-cluster-select-method">
              <Select
                defaultValue={cluster}
                style={{ width: 90 }}
                size="small"
                onChange={handleClusterChange}
                options={[
                  {
                    value: "kmeans",
                    label: "KMeans",
                  },
                  {
                    value: "gmm",
                    label: "GMM",
                  },
                ]}
              />
            </div>
          </div>

          <div className="control-cluster-switch-all">
            <div className="control-cluster-switch">
              <Radio.Group
                onChange={handleDimReductionChange}
                value={dimReduction}
                size="small"
              >
                <Radio value={"umap"}>UMAP</Radio>
                <Radio value={"tsne"}>t-SNE</Radio>
              </Radio.Group>
            </div>
          </div>
        </div>
      </div>

      <div className="control-lines">
        <div className="control-name">Lines</div>
        <div className="control-lines-select">
          <Radio.Group
            onChange={handleLineGroupTypeChange}
            value={lineGroupType}
            size="small"
          >
            <Radio value={"sensor"}>Sensor</Radio>
            <Radio value={"cluster"}>Cluster</Radio>
          </Radio.Group>
        </div>
      </div>

      {/* <div className="control-mul-pollutant">
        <div className="control-name">BoxPlot</div>
        <div className="control-mul-pollutant-select">
          <Select
            defaultValue={rangeBarList}
            mode="multiple"
            allowClear
            style={{ width: "100%" }}
            size="small"
            onChange={handleRangeBarListChange}
            options={POLLUTANTLIST_25}
          />
        </div>
      </div> */}

      <div className="control-mul-pollutant">
        <div className="control-name">Groups</div>
        <div className="control-mul-pollutant-select">
          <Select
            defaultValue={groupLineList}
            mode="multiple"
            allowClear
            style={{ width: "100%" }}
            size="small"
            onChange={handleGroupLineListChange}
            options={POLLUTANTLIST_25}
          />
        </div>
      </div>

      <div className="control-lines">
        <div className="control-name">Task</div>
        <div className="control-lines-select">
          <Radio.Group
            onChange={(e)=>{handleTaskTypeChange(e.target.value)}}
            defaultValue={taskType}
            size="small"
          >
            <Radio value={"anomaly"}>Anomaly</Radio>
            <Radio value={"feature"}>Feature</Radio>
          </Radio.Group>
        </div>
      </div>
    </div>
  );
};

export default ControlBar;
