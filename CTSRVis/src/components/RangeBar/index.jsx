import React, { useEffect, useRef, useState } from "react";
import * as echarts from "echarts";

import { getRangeBar } from "../../apis/api";
import { POLLUTANTLIST_NAME } from "../../utils/constants";
import "./index.less";

const RangeBar = (props) => {
  const { rangeBarList, rangeBarTimeStampIdx, rangeBarPollutantIdx } = props;
  const [data, setData] = useState([]);
  const [dataP, setDataP] = useState([]);
  const [pollu, setPollu] = useState([]);
  const [times, setTimes] = useState([]);
  const [chartData, setChartData] = useState([]);
  const [chartPollu, setChartPollu] = useState([]);

  const rangeBarRef = useRef(null);

  const drawRangeBar = () => {
    if (rangeBarRef.current === null) return;
    let existInstance = echarts.getInstanceByDom(rangeBarRef.current);
    if (existInstance !== undefined) {
      echarts.dispose(existInstance);
    }

    let colorMax = chartData.map(Function.apply.bind(Math.max, null));
    colorMax = Math.max.apply(null, colorMax);

    let colorMax_bottom = dataP.map(Function.apply.bind(Math.max, null));
    colorMax_bottom = Math.max.apply(null, colorMax_bottom);

    const barHeight = colorMax * 0.01;

    const rangeChart = echarts.init(rangeBarRef.current);
    const option = {
      visualMap: [
        {
          show: false,
          type: "continuous",
          seriesIndex: 1,
          max: colorMax,
          min: 0,
        },
        {
          show: false,
          type: "continuous",
          seriesIndex: 3,
          max: 7000,
          min: 0,
        },
        {
          show: false,
          type: "continuous",
          seriesIndex: 5,
          max: colorMax,
          min: 0,
        },
        {
          show: false,
          type: "continuous",
          seriesIndex: 7,
          max: 7000,
          min: 0,
        },
      ],
      legend: {
        show: false,
        top: "bottom",
        data: ["Range", "Average"],
      },
      grid: [
        {
          top: "3%",
          left: "5%",
          right: "1%",
          height: "38%",
        },
        {
          left: "5%",
          right: "1%",
          top: "53%",
          height: "38%",
        },
      ],
      xAxis: [
        {
          type: "category",
          data: chartPollu,
        },
        {
          type: "category",
          data: times,
          gridIndex: 1,
        },
      ],
      yAxis: [
        {
          type: "value",
          min: 0,
          max: colorMax,
        },
        {
          type: "value",
          min: 0,
          max: colorMax_bottom, // TODo
          gridIndex: 1,
        },
      ],
      dataZoom: [
        {
          type: "inside",
          xAxisIndex: [0, 1],
          start: 0,
          end: 100,
        },
        // {
        //   type: "slider",
        //   start: 0,
        //   end: 100,
        //   xAxisIndex: [0, 1],
        //   height: 22,
        //   bottom: 12,
        // },
      ],
      tooltip: {
        show: true,
        trigger: "axis",
      },
      series: [
        {
          type: "bar",
          itemStyle: {
            color: "transparent",
          },
          data: chartData.map(function (d) {
            return d[0];
          }),
          coordinateSystem: "cartesian2d",
          stack: "Min Max",
          silent: true,
          xAxisIndex: 0,
          yAxisIndex: 0,
        },
        {
          type: "bar",
          data: chartData.map(function (d) {
            return d[1] - d[0];
          }),
          coordinateSystem: "cartesian2d",
          name: "Range",
          stack: "Min Max",
          xAxisIndex: 0,
          yAxisIndex: 0,
        },
        {
          type: "bar",
          itemStyle: {
            color: "transparent",
          },
          data: chartData.map(function (d) {
            return d[2] - barHeight;
          }),
          coordinateSystem: "cartesian2d",
          stack: "Average",
          silent: true,
          z: 10,
          xAxisIndex: 0,
          yAxisIndex: 0,
        },
        {
          type: "bar",
          data: chartData.map(function (d) {
            return barHeight * 2;
          }),
          coordinateSystem: "cartesian2d",
          name: "Average",
          stack: "Average",
          barGap: "-100%",
          z: 10,
          xAxisIndex: 0,
          yAxisIndex: 0,
        },
        {
          type: "bar",
          itemStyle: {
            color: "transparent",
          },
          data: dataP.map(function (d) {
            return d[0];
          }),
          coordinateSystem: "cartesian2d",
          stack: "Min Max",
          silent: true,
          xAxisIndex: 1,
          yAxisIndex: 1,
        },
        {
          type: "bar",
          data: dataP.map(function (d) {
            return d[1] - d[0];
          }),
          coordinateSystem: "cartesian2d",
          name: "Range",
          stack: "Min Max",
          xAxisIndex: 1,
          yAxisIndex: 1,
        },
        {
          type: "bar",
          itemStyle: {
            color: "transparent",
          },
          data: dataP.map(function (d) {
            return d[2] - barHeight;
          }),
          coordinateSystem: "cartesian2d",
          stack: "Average",
          silent: true,
          z: 10,
          xAxisIndex: 1,
          yAxisIndex: 1,
        },
        {
          type: "bar",
          data: dataP.map(function (d) {
            return barHeight * 2;
          }),
          coordinateSystem: "cartesian2d",
          name: "Average",
          stack: "Average",
          barGap: "-100%",
          z: 10,
          xAxisIndex: 1,
          yAxisIndex: 1,
        },
      ],
    };
    rangeChart.setOption(option, true);
    window.onresize = rangeChart.resize;
  };

  useEffect(() => {
    getRangeBar(rangeBarTimeStampIdx, rangeBarPollutantIdx).then((res) => {
      let { data, pollutant, dataP, times } = res;
      setData([...data]);
      setPollu([...pollutant]);
      setDataP([...dataP]);
      setTimes([...times]);
    });
  }, [rangeBarTimeStampIdx, rangeBarPollutantIdx]);

  useEffect(() => {
    if (data.length === 0) return;
    let idxes = rangeBarList.map((item) => POLLUTANTLIST_NAME.indexOf(item));

    if (rangeBarList.length !== 0) {
      let cd = idxes.map((item) => data[item]);
      let cp = idxes.map((item) => pollu[item]);
      setChartData(cd);
      setChartPollu(cp);
    } else {
      setChartData(data);
      setChartPollu(pollu);
    }
  }, [rangeBarList, data, pollu]);

  useEffect(() => {
    if (chartData.length !== 0) {
      drawRangeBar();
    }
  }, [chartData]);

  return <div className="range-bar" ref={rangeBarRef}></div>;
};

export default RangeBar;
