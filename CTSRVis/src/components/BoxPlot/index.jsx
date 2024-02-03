import { useEffect, useRef } from "react";
import * as echarts from "echarts";

import { getBoxPlot } from "../../apis/api";
import "./index.less";
import { POLLUTANTLIST_NAME } from "../../utils/constants";

const BoxPlot = (props) => {
  const { rangeBarList, rangeBarTimeStampIdx, rangeBarPollutantIdx } = props;

  const boxPlotTsRef = useRef(null);
  const boxPlotPoRef = useRef(null);

  const drawBoxPlot = (data, charRef, indicator) => {
    let existInstance = echarts.getInstanceByDom(charRef.current);
    if (existInstance !== undefined) {
      echarts.dispose(existInstance);
    }
    const drawBoxChart = echarts.init(charRef.current);
    const option = {
      title: [
        {
          subtext: indicator === "ts" ? "at timestamp" : "on pollutant",
          left: "5%",
        },
      ],
      dataset: [
        {
          source: data,
        },
        {
          transform: {
            type: "boxplot",
            config: { itemNameFormatter: "{value}" },
          },
        },
        {
          fromDatasetIndex: 1,
          fromTransformResult: 1,
        },
      ],
      tooltip: {
        trigger: "item",
        axisPointer: {
          type: "shadow",
        },
      },
      grid: {
        left: "5%",
        right: "5%",
        top: "8%",
        bottom: "16%",
      },
      xAxis: {
        type: "category",
        boundaryGap: true,
        nameGap: 30,
        splitArea: {
          show: false,
        },
        splitLine: {
          show: false,
        },
      },
      yAxis: {
        type: "value",
        name: "ug/m3",
        splitArea: {
          show: true,
        },
      },
      dataZoom: [
        {
          type: "inside",
          start: indicator === "ts" ? 0 : 5,
          end: indicator === "ts" ? 100 : 25,
        },
      ],
      series: [
        {
          name: "boxplot",
          type: "boxplot",
          datasetIndex: 1,
        },
        {
          name: "outlier",
          type: "scatter",
          datasetIndex: 2,
        },
      ],
    };

    drawBoxChart.setOption(option, true);
  };

  useEffect(() => {
    getBoxPlot(rangeBarTimeStampIdx, rangeBarPollutantIdx).then((res) => {
      const { dataP, dataT } = res;
      let idxes = rangeBarList.map((item) => POLLUTANTLIST_NAME.indexOf(item));

      if (rangeBarList.length !== 0) {
        let datat = idxes.map((item) => dataT[item]);
        // let datap = idxes.map((item) => dataP[item]);

        drawBoxPlot(datat, boxPlotTsRef, "ts");
        drawBoxPlot(dataP, boxPlotPoRef, "po");
      } else {
        drawBoxPlot(dataT, boxPlotTsRef, "ts");
        drawBoxPlot(dataP, boxPlotPoRef, "po");
      }
      //   console.log("boxplot", res);
    });
  }, [rangeBarList, rangeBarTimeStampIdx, rangeBarPollutantIdx]);

  return (
    <div className="box-plot">
      <div className="box-plot-po" ref={boxPlotPoRef}></div>
      <div className="box-plot-ts" ref={boxPlotTsRef}></div>
    </div>
  );
};

export default BoxPlot;
