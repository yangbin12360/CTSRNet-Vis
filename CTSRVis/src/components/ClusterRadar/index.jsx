import { useEffect, useRef } from "react";
import * as echarts from "echarts";

import "./index.less";

const ClusterRadar = (props) => {
  let { series, maxes, name } = props;
  series = series.map((item) => parseFloat(item.toFixed(2)));

  const indis = maxes.map((item, index) => {
    return { text: name[index], max: item };
  });
  const radarRef = useRef(null);

  useEffect(() => {
    drawRadar(radarRef);
  }, [series]);

  const drawRadar = (radarRef) => {
    let existInstance = echarts.getInstanceByDom(radarRef.current);
    if (existInstance !== undefined) {
      echarts.dispose(existInstance);
    }

    const radarChart = echarts.init(radarRef.current);
    const option = {
      tooltip: {
        trigger: "axis",
        confine: true,
        padding: 5,
        extraCssText: "width:100px;",
      },
      radar: [
        {
          nameGap:5,
          axisName: {
            show: true,
            fontSize:10,
            fontFamily:'Arial',
           color:'#7b7d85'
          },
          indicator: indis,
          center: ["50%", "65%"],
          radius: 87 / series.length - 8,
        },
      ],
      series: [
        {
          type: "radar",
          tooltip: {
            trigger: "item",
          },
          symbolSize: 4,
          data: [
            {
              value: series,
              name: "Scores",
            },
          ],
          color: "#70a178",
        },
      ],
    };
    radarChart.setOption(option);
    window.onresize = radarChart.resize;
  };

  return <div className="cluster-radar" ref={radarRef}></div>;
};

export default ClusterRadar;
