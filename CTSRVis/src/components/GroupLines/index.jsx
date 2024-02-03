import React, { useEffect, useRef, useState } from "react";
import { Radio } from "antd";
import * as echarts from "echarts";

import { getGroupLinesCluster, getGroupLinesSensor } from "../../apis/api";
import {
  POLLUTANTLIST_NAME,
  COLORLIST,
  SENSORLIST_NAME,
  SENSORLIST_NAME_EN1
} from "../../utils/constants";
import "./index.less";

const GroupLines = (props) => {
  const {
    lineGroupType,
    clusterLabel,
    groupLineList,
    setRangeBarPollutantIdx,
    setRangeBarTimeStampIdx,
  } = props;
  // console.log("clusterLabel", clusterLabel);
  const groupLinesRef = useRef(null);
  const [isZoomForAll, setIsZoomForAll] = useState("unify");
  const dataZoom = useRef(null);

  useEffect(() => {
    if (groupLinesRef.current != null && clusterLabel.length !== 0) {
      if (lineGroupType === "sensor") {
        getGroupLinesSensor().then((res) => {
          initSensorGraph(res);
        });
      } else if (lineGroupType === "cluster") {
        getGroupLinesCluster(clusterLabel, groupLineList).then((res) => {
          initClusterGraph(res);
        });
      }
    }
  }, [lineGroupType, groupLineList, isZoomForAll, clusterLabel]);

  const initSensorGraph = (res) => {
    console.log("ressssssssss",res);
    if (groupLinesRef.current === null) return;
    let { data, label, times } = res;

    let idxes = groupLineList.map((item) => POLLUTANTLIST_NAME.indexOf(item));

    if (groupLineList.length !== 0) {
      data = idxes.map((item) => data[item]);
      label = idxes.map((item) => label[item]);
    } else {
      data = data.slice(0, 3);
      label = label.slice(0, 3);
    }

    const title = [];
    const xAxis = [];
    const yAxis = [];
    const series = [];
    const grid = [];
    const chartHeight = 90;
    groupLinesRef.current.style.height = `${
      (chartHeight + 44) * label.length
    }px`;
    label.forEach((item, index) => {
      title.push({
        textBaseline: "middle",
        top: 22 + (32 + chartHeight) * index,
        left: "5%",
        subtext: item,
      });
      grid.push({
        left: "5%",
        right: "1%",
        height: chartHeight,
        top: 22 + (32 + chartHeight) * index,
      });
      xAxis.push({
        data: times,
        type: "category",
        gridIndex: index,
      });
      yAxis.push({
        gridIndex: index,
      });

      data[index].forEach((d, i) => {
        series.push({
          name: SENSORLIST_NAME_EN1[i],
          type: "line",
          data: d,
          xAxisIndex: index,
          yAxisIndex: index,
          smooth: true, //是否平滑
          markArea: {
            silent: true,
            itemStyle: {
              color: "#ddd",
              opacity: 0.1,
            },
            data: [
              [
                {
                  xAxis: "20170103",
                },
                {
                  xAxis: "20171227",
                },
              ],
              [
                {
                  xAxis: "20190103",
                },
                {
                  xAxis: "20191227",
                },
              ],
              [
                {
                  xAxis: "20210105",
                },
                {
                  xAxis: "20211224",
                },
              ],
            ],
          },
        });
      });
    });
    const option = {
      title: title,
      tooltip: {
        trigger: "axis",
        formatter: function (params) {
          let tip = params.map((item, index) => {
            return (
              item.marker +
              SENSORLIST_NAME_EN1[item.componentIndex % 10] +
              " " +
              item.value
            );
          });
          let result = params[0].name + "<br/>";
          for (let i of tip) {
            result += i + "<br/>";
          }
          return result;
        },
      },
      grid: grid,
      xAxis: xAxis,
      yAxis: yAxis,
      series: series,
      color: COLORLIST.slice(0, 10),
      legend: {
        show: true,
        top: 0,
        left: "10%",
      },
      dataZoom: [
        {
          type: "inside",
          xAxisIndex: [0, 1, 2,3,4,5,6,7,8,9],
          start: 0,
          end: 100,
        },
        {
          type: "slider",
          start: 0,
          end: 100,
          xAxisIndex: [0, 1, 2,3,45,6,7,8,9],
          height: 30,
          bottom:10 ,
          right:25
        },
      ],
    };

    let existInstance = echarts.getInstanceByDom(groupLinesRef.current);
    if (existInstance !== undefined) {
      echarts.dispose(existInstance);
    }

    const myChart = echarts.init(groupLinesRef.current);
    myChart.setOption(option, true);
    // 返回时间
    myChart.on("click", (e) => {
      const timeStamp = e.name;
      let sIndex = parseInt(e.seriesIndex / 10);
      let polluIndex = POLLUTANTLIST_NAME.indexOf(label[sIndex]);
      setRangeBarTimeStampIdx(timeStamp);
      setRangeBarPollutantIdx(polluIndex);
    });
  };

  const initClusterGraph = (res) => {
    if (groupLinesRef.current === null) return;
    let { data, times, max } = res;

    console.log("Res", res);
    const title = [];
    const xAxis = [];
    const yAxis = [];
    const series = [];
    const grid = [];
    const chartHeight = 89;
    const chartWidth = 160;

    if (isZoomForAll === "unify") {
      dataZoom.current = [
        {
          type: "inside",
          xAxisIndex: Array.from(Array(30).keys()),
          start: 0,
          end: 100,
        },
        {
          type: "slider",
          xAxisIndex: Array.from(Array(30).keys()),
          start: 0,
          end: 100,
          top: 0,
          left: "28%",
          height: "20px",
        },
      ];
    } else {
      let dz = [];
      for (let i = 0; i < 30; i++) {
        dz.push({
          type: "inside",
          xAxisIndex: [i],
          start: 0,
          end: 100,
        });
      }
      dataZoom.current = dz;
    }

    data.forEach((lineData, line) => {
      // 每行数据
      lineData.forEach((gridData, column) => {
        //处理首列
        if (column === 0) {
          title.push({
            textBaseline: "middle",
            top: 30 + (chartHeight + 40) * line,
            left: "4%",
            subtext: groupLineList[line],
          });
          yAxis.push({
            gridIndex: line * 5 + column,
            splitLine: {
              show: false,
            },
            min: 0,
            max: max[line],
            splitArea: { show: true },
          });
        } else {
          title.push({
            show: false,
          });
          yAxis.push({
            gridIndex: line * 5 + column,
            // show: false,
            axisLabel: { show: false },
            min: 0,
            max: max[line],
            splitArea: { show: true },
          });
        }
        grid.push({
          left: column * (chartWidth + 24) + 40,
          top: line * (chartHeight + 36) + 40,
          height: chartHeight,
          width: chartWidth,
        });
        xAxis.push({
          data: times[column].map((item) => item.slice(4)),
          type: "category",
          gridIndex: line * 5 + column,
        });
        gridData.forEach((data, index) => {
          series.push({
            name: "Cluster " + index,
            type: "line",
            data: data,
            xAxisIndex: line * 5 + column,
            yAxisIndex: line * 5 + column,
            smooth: true,
          });
        });
      });
    });
    const option = {
      title: title,
      tooltip: {
        trigger: "axis",
        formatter: function (params) {
          let tip = params.map((item, index) => {
            return (
              item.marker +
              "Cluster" +
              (item.componentIndex % 10) +
              " " +
              item.value.toFixed(2)
            );
          });
          let result = params[0].name + "<br/>";
          for (let i of tip) {
            result += i + "<br/>";
          }
          return result;
        },
      },
      grid: grid,
      xAxis: xAxis,
      yAxis: yAxis,
      series: series,
      color: COLORLIST.slice(0, Math.max(...clusterLabel) + 1),
      dataZoom: dataZoom.current,
      legend: {
        show: true,
        left: "48%",
        top: 0,
      },
    };
    let existInstance = echarts.getInstanceByDom(groupLinesRef.current);
    if (existInstance !== undefined) {
      echarts.dispose(existInstance);
    }
    const myChart = echarts.init(groupLinesRef.current);
    myChart.setOption(option, true);
  };

  return (
    <div className="group-lines">
      {lineGroupType === "cluster" && (
        <div className="group-lines-bar">
          <Radio.Group
            value={isZoomForAll}
            onChange={(e) => {
              setIsZoomForAll(e.target.value);
            }}
            size="small"
          >
            <Radio value={"indiv"}>Zoom for one</Radio>
            <Radio value={"unify"}>Zoom for all</Radio>
          </Radio.Group>
        </div>
      )}
      <div
        id="chart-line"
        style={{
          width: "100%",
          height: "900px",
          overflowY: "scroll",
        }}
      >
        <div style={{ width: "100%" }} ref={groupLinesRef}></div>
      </div>
    </div>
  );
};

export default GroupLines;
