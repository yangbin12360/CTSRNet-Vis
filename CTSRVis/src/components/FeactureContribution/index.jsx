import * as echarts from "echarts";
import { useEffect, useRef } from "react";
import { POLLUTANTLIST_25 } from "../../utils/constants";

import "./index.less";

const FeatureContribution = (props) => {
  const { fcData } = props;
  const fcsDataRef = useRef(null);
  const xAxisListRef = useRef(null);
  const yAxisListRef = useRef(null);

  const fcRef = useRef(null);

  const drawFC = (k_cluster, symbolSize) => {
    if (fcRef.current === null) return;
    let existInstance = echarts.getInstanceByDom(fcRef.current);
    if (existInstance !== undefined) {
      echarts.dispose(existInstance);
    }
    const fcChart = echarts.init(fcRef.current);
    const option = {
      tooltip: {
        position: "top",
        formatter: function (params) {
          return (
            "FC of cluster " +
            params.value[0] +
            " of Dimension " +
            params.value[1] +
            ": " +
            params.value[2].toFixed(2)
          );
        },
      },
      grid: {
        top: "5%",
        left: "2%",
        bottom: 10,
        width: symbolSize * (k_cluster ),
        containLabel: true,
      },
      xAxis: {
        type: "category",
        data: yAxisListRef.current,
        boundaryGap: false,
        splitLine: {
          show: true,
        },
        axisLine: {
          show: false,
        },
        axisLabel: {
          formatter: "Cluster {value}",
          align: "left",
        },
      },
      yAxis: {
        name: "Dimension",
        type: "category",
        data: xAxisListRef.current.map(
          (item, index) => POLLUTANTLIST_25[index]
        ),
        axisLine: {
          show: false,
        },
      },
      visualMap: {
        show: true,
        type: "continuous",
        min: -0.6,
        max: 0.6,
        inRange: {
          color: ["#4580ae", "#88c3dd", "#dadada", "#ff9242", "#bf4527"],
        },
        right: "2%",
        bottom: "6%",
        text: ["HIGH", "LOW"],
      },
      series: [
        {
          name: "Punch Card",
          type: "scatter",
          symbol: "rect",
          symbolSize: [symbolSize * 0.75, 20],
          symbolOffset: [(symbolSize * 0.75) / 2, 0],
          data: fcsDataRef.current,
          animationDelay: function (idx) {
            return idx * 5;
          },
        },
      ],
    };

    fcChart.setOption(option, true);
  };

  useEffect(() => {
    if (fcData) {
      let k_cluster = fcData.length;
      let symbolSize = parseInt(560 / k_cluster);
      let drawFcData = fcData.map((item, index) => {
        return item.map((it, id) => {
          return [index, id, it];
        });
      });
      xAxisListRef.current = Array.from(Array(fcData[0].length).keys());
      yAxisListRef.current = Array.from(Array(fcData.length).keys());
      fcsDataRef.current = drawFcData.flat();
      drawFC(k_cluster, symbolSize);
    }
  }, [fcData]);
  return <div className="fc" ref={fcRef}></div>;
};

export default FeatureContribution;
