import { useEffect, useRef } from "react";
import * as echarts from "echarts";

import { COLORLIST } from "../../utils/constants";
import "./index.less";

const ClusterBar = (props) => {
  const { series, index } = props;

  const series_bar = series.map((item, index) => {
    return {
      name: `Cluster ${index}`,
      type: "bar",
      stack: "total",
      emphasis: {
        focus: "series",
      },
      data: [item],
    };
  });

  const cBarRef = useRef(null);

  useEffect(() => {
    drawRadar(cBarRef);
  }, [series]);

  const drawRadar = (cBarRef) => {
    let existInstance = echarts.getInstanceByDom(cBarRef.current);
    if (existInstance !== undefined) {
      echarts.dispose(existInstance);
    }

    const cBarChart = echarts.init(cBarRef.current);
    const option = {
      tooltip: {
        trigger: "axis",
        axisPointer: {
          type: "shadow",
        },
        confine: true,
        padding: 2,
        extraCssText: "width:120px;",
      },
      xAxis: {
        type: "value",
        show: false,
      },
      yAxis: {
        type: "category",
        show: false,
      },
      color: COLORLIST,
      series: series_bar,
    };
    cBarChart.setOption(option);

    window.onresize = cBarChart.resize;
  };

  return (
    <div
      className="cluster-bar"
      id={`bar-list-item-${index}`}
      ref={cBarRef}
    ></div>
  );
};

export default ClusterBar;
