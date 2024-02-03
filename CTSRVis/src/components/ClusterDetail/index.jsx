import { useEffect, useRef } from "react";
import * as echarts from "echarts";

import { COLORLIST } from "../../utils/constants";
import "./index.less";

const ClusterDetail = (props) => {
  const { detail, title, cIndex } = props;
  const series = detail[cIndex - 1];
  const detailRef = useRef(null);
  console.log(series,"ssssss");
  const drawDetail = () => {
    let existInstance = echarts.getInstanceByDom(detailRef.current);
    if (existInstance !== undefined) {
      echarts.dispose(existInstance);
    }
    const detailChart = echarts.init(detailRef.current);
    const option = {
      tooltip: {
        trigger: "axis",
        axisPointer: {
          type: "shadow",
        },
        formatter: function (params) {
          let tip =
            "Cluster " +
            params[0].seriesIndex +
            "<br/>" +
            params[0].marker +
            params[0].value.toFixed(3);

          return tip;
        },
      },
      title: {
        subtext: "cluster " + title,
        top: "-12%",
        left: "8%",
      },
      xAxis: {
        type: "category",
      },
      yAxis: {
        type: "value",
      },
      grid: [
        {
          top: "12%",
          bottom: "16%",
          height: "60%",
        },
      ],
      series: [
        {
          data: series,
          type: "bar",
          showBackground: true,
          itemStyle: {
            color: (params) => {
              return COLORLIST[params.dataIndex];
            },
          },
          backgroundStyle: {
            color: "rgba(180, 180, 180, 0.2)",
          },
        },
      ],
    };

    detailChart.setOption(option, true);

    window.onresize = detailChart.resize;
  };

  useEffect(() => {
    drawDetail(detailRef);
    console.log("deta:",detail,cIndex);
  }, [detail, cIndex]);

  return <div className="cluster-detail" ref={detailRef}></div>;
};

export default ClusterDetail;
