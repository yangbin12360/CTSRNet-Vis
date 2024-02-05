import { useEffect, useState, useRef } from "react";
import * as echarts from "echarts";

import { getFeatureContribution } from "../../apis/api";

import { COLORLIST } from "../../utils/constants";
import "./index.less";

const FeatureTransition = (props) => {
  const { dataset, clusterLabel, setFcData} = props;

  const [featureTimestamp, setFeatureTimeStamp] = useState(0); // 特征分析视图time stamp在61长度上对应的idx
  const [ftData, setFtData] = useState(null); // FT 视图数据

  const transRef = useRef(null);
  const isSelf = useRef(false);
  // let ft = startFtData;
  // let fc = startFcData;
  // setFtData(ft);
  // setFcData(fc);
  const generateData = () => {
    const data = [];
    for (let i = 0; i < 6; i++) {
      const innerArray = [];
      for (let j = 0; j < 61; j++) {
        let randomNum = Math.random() * 1.8 - 0.9; // 生成[-1.2, 1.2]之间的浮点数
        randomNum = parseFloat(randomNum.toFixed(2)); // 限制到小数点后两位，并转回为数字
        innerArray.push(randomNum);
      }
      data.push(innerArray);
    }
    return data;
  };
  const generateFcData = () => {
    const data = [];
    for (let i = 0; i < 5; i++) {
      const innerArray = [];
      for (let j = 0; j < 25; j++) {
        let randomNum = Math.random() * 2.4 - 1.2; // 生成[-1.2, 1.2]之间的浮点数
        randomNum = parseFloat(randomNum.toFixed(2)); // 限制到小数点后两位，并转回为数字
        innerArray.push(randomNum);
      }
      data.push(innerArray);
    }
    return data;
  };
  const drawLines = () => {
    let existInstance = echarts.getInstanceByDom(transRef.current);
    if (existInstance !== undefined) {
      echarts.dispose(existInstance);
    }

    const series = [];

    ftData.forEach((d, i) => {
      series.push({
        name: "Cluster" + i,
        type: "line",
        data: d,
        step: "middle",
        smooth: true, //是否平滑
      });
    });

    const option = {
      tooltip: {
        trigger: "axis",
        formatter: function (params) {
          let tip = params[0].name + "<br/>";
          let tips = params.map((item) => {
            return (
              item.marker +
              item.seriesName +
              " " +
              item.value.toFixed(3) +
              "<br/>"
            );
          });
          tips.forEach((item) => {
            tip += item;
          });
          return tip;
        },
      },
      grid: {
        left: "5%",
        right: "5%",
        bottom: "22%",
        top: "12%",
      },
      xAxis: {
        data: Array.from(Array(61).keys()),
        type: "category",
      },
      yAxis: {
        type: "value",
        name: "FC",
      },
      series: series,
      color: COLORLIST,
      dataZoom: [
        {
          type: "inside",
          start: 0,
          end: 100,
        },
        {
          type: "slider",
          start: 0,
          end: 100,
          height: 22,
          bottom: 10,
        },
      ],
      legend: {
        show: true,
      },
    };

    const transChart = echarts.init(transRef.current);
    transChart.setOption(option, true);

    transChart.on("click", (e) => {
      setFeatureTimeStamp(e.dataIndex);
      isSelf.current = true;
    });
  };

  useEffect(()=>{
    let newFt = caseFtData;
    let newFc = caseFcData;
    setFcData(newFc)
    setFtData(newFt)
  },[clusterLabel,])


  useEffect(() => {
    if (clusterLabel.length !== 0) {
      let ft = startFtData;
      let fc = startFcData;
      if (!isSelf.current) {
        // console.log(ft);
        setFtData(ft);
      }
      setFcData(fc);
      // console.log(fc);
      isSelf.current = false;
    }
  }, [ featureTimestamp]);

  useEffect(() => {
    if (ftData) {
      drawLines();
    }
  }, [ftData]);


  return <div className="trans" ref={transRef}></div>;
};

export default FeatureTransition;
