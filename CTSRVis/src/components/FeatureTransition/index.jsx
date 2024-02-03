import { useEffect, useState, useRef } from "react";
import * as echarts from "echarts";

import { getFeatureContribution } from "../../apis/api";

import { COLORLIST } from "../../utils/constants";
import "./index.less";

const caseFtData = [
  [
      0.38,
      -0.27,
      0.05,
      0.04,
      -0.62,
      0.05,
      0.39,
      0.48,
      0.14,
      -0.25,
      -0.18,
      0.25,
      0.17,
      -0.34,
      0.68,
      0.29,//45
      0.15,
      0.3,
      0.58,
      0.45,
      0.7,//40
      0.09,
      0.32,
      0.35,
      0.61,
      0.87,//35
      0.58,
      -0.15,
      0.1,
      0.6,
      0.55,//30
      0.6,
      0.1,
      -0.15,
      0.53,
      0.87,//34
      0.61,
      0.35,
      0.32,
      0.45,//39
      0.7,//40
      0.45,
      0.58,
      0.3,
      0.15,
      0.29,//45
      0.7,//46
      -0.86,
      -0.17,
      -0.24,
      -0.55,
      -0.43,
      -0.69,
      0.89,
      0.15,
      -0.69,
      -0.1,
      -0.28,
      -0.23,
      0.53,
      0.32
  ],
  [
      -0.57,
      0.35,
      0.73,
      0.73,
      -0.42,
      0.04,
      -0.45,
      -0.23,
      0.68,
      0.53,
      -0.25,
      -0.7,
      0.21,
      0.58,
      0.42,
      -0.34,
      -0.55,
      0.41,
      -0.02,
      0.61,
      -0.49,
      0.04,
      0.06,
      -0.58,
      -0.4,
      0.71,
      -0.84,
      -0.25,
      -0.25,
      0.34,
      0.08,//30
      -0.02,
      0.1,
      0.3,
      -0.45,
      0.03,//35
      0.01,
      -0.03,
      -0.57,//38
      -0.54,//39
      -0.61,
      -0.57,//41
      -0.5,//42
      0.07,//43
      -0.07,//44
      -0.32,
      -0.36,//46
      -0.08,
      0.68,
      -0.48,
      -0.87,
      0.57,
      0.19,
      0.62,
      0.45,
      -0.36,
      0.11,
      -0.1,
      0.1,
      0.7,
      0.73
  ],
  [
      -0.07,
      -0.03,
      0.17,
      0.56,
      0.57,
      -0.4,
      0.84,
      0.84,
      0.19,
      -0.07,
      -0.12,
      0.47,
      -0.45,
      -0.83,
      -0.22,
      -0.12,
      0.73,
      0.34,
      -0.15,
      0.2,
      -0.51,
      -0.78,
      -0.31,
      0.12,
      0.43,
      -0.43,
      -0.83,
      0.41,
      0.69,
      0,
      0.31,//30
      0.1,
    -0.31,
      -0.31,
      0.62,
      0.4,//35
      0.35,
      -0.29,
      0.4,
      0.45,//39
      0.72,
      0.45,//41
      0.58,
      -0.3,
      -0.26,
      0.5,
      0.85,//46
      -0.38,
      -0.09,
      -0.47,
      -0.07,
      0.03,
      -0.15,
      -0.86,
      -0.22,
      0.3,
      -0.32,
      0.2,
      0.69,
      -0.87,
      0.87
  ],
  [
      -0.8,
      -0.32,
      -0.29,
      -0.64,
      0.73,
      0.54,
      -0.07,
      -0.52,
      0.08,
      0.64,
      0.1,
      0.22,
      -0.49,
      -0.47,
      -0.35,
      0.7,
      0.47,
      -0.5,
      -0.41,
      0.28,
      -0.81,
      -0.42,
      0.53,
      -0.71,
      -0.47,
      0.74,
      0.78,
      0.85,
      -0.9,
      -0.89,
      0.53,//30
      0.5,
      0.3,
      0.15,
      0.5,
      0.62,
      0.62,
      0.3,
      0.27,
      0,//39
      0.34,
      -0.23,
      0.57,
      0.4,
      0.2,
      0.32,
      0.5,//46
      -0.26,
      0.36,
      0.36,
      -0.73,
      -0.64,
      -0.7,
      -0.04,
      0.04,
      -0.14,
      -0.04,
      -0.43,
      0.76,
      -0.15,
      -0.76
  ],
  [
    0.85,
    0.22,
    0.05,
    -0.61,
    0.01,
    0.19,
    0.44,
    -0.66,
    0.41,
    -0.24,
    -0.28,
    0.44,
    -0.85,
    0.08,
    -0.25,
    0.6,
    -0.09,
    -0.66,
    0.83,
    -0.66,
    -0.16,
    0.71,
    0.87,
    -0.8,
    -0.86,
    0.53,
    -0.78,
    0.47,
    0.47,
    0.8,
    0.1,//30
    -0.02,
    0.1,
    0.3,
    -0.34,//34
    -0.33,
    -0.59,
    -0.04,
    -0.41,//38
    0.29,
    -0.5,
    -0.23,
    -0.61,
    -0.7,
    0.29,
    0.41,
    -0.05,//46
    0.86,
    -0.81,
    0.34,
    0.22,
    0.07,
    0.6,
    -0.76,
    -0.3,
    -0.69,
    -0.46,
    -0.79,
    -0.04,
    -0.53,
    0.4
],
  [
      -0.54,
      -0.42,
      0.37,
      0.71,
      -0.89,
      0.47,
      -0.75,
      -0.5,
      -0.42,
      0.28,
      0.36,
      0.36,
      0.82,
      -0.31,
      -0.11,
      0.52,
      0.79,
      -0.58,
      -0.55,
      -0.18,
      0.56,
      -0.47,
      -0.48,
      0.6,
      -0.39,
      0.61,
      0.13,
      0.57,
      -0.37,
      0.3,
      -0.1,//30
      -0.02,
      -0.58,
      -0.9,
      0.22,
      0.5,
      0.3,
      0.04,
      0.02,
      0.17,//39
      0.5,
      0.53,
      0.28,//42
      0.13,
      -1.05,
      -0.5,
      0.28,//46
      0.6,
      -0.22,
      -0.47,
      -0.67,
      0.34,
      -0.38,
      -0.11,
      0.04,
      0.02,
      -0.88,
      -0.02,
      -0.21,
      -0.87,
      0.73
  ]
]
const caseFcData = [
  [
    0.89, -0.13, 0.8, 0.55, 0.67, 0.55,-0.13, 0.55, 0.75, 0.5, 0.55, -0.2,
    0.02, 0.49, 0.49, 0.42, 0.84, -0.3, 0.84, 0, -0.35, -0.2, -0.3, 0.77,
    0.77,
  ],
  [
    0.2, -0.11, 0.3, 0.13, 0.13, 0.3,0.15 , -0.3, 0.3, 0, 0.45, -0.39,
    -0.35, 0.23, -0.3, -0.16, 0.23, -0.39, 0,-0.25, 0, -0.05, -0.35,
    0.23, 0.29,
  ],
  [
    -0.3, 0.06, -0.04, -0.04, -0.04, -0.04, 0.23, 0.23, -0, 0.22, 0.1, 0,
    0.35, 0.03, -0.18, -0.03, -0.03, -0.23, 0.07, -0.1, -0.25, 0.15, 0.15, 0.2,
    -0.1,
  ],
  [
    0.1, 0.2, 0.1, -0.1, 0.1, 0, 0.82, 0.34, 0.13, -0.05, 0.05, 0.34,
    0.05, 0.35, 0, 0, 0.03, 0.32, 0.1, 0.14, 0.29, 0.2, -0.12, 0.32,
    0,
  ],
  [
    -0.3, 0.49, 0.31, -0.2, 0.23, 0.17, 0.55, -0, -0.2, 0, -0.2,
    -0.7, -0.2,-0.15, 0.02, -0.2, 0.01, -0.37, 0.12, -0.5, 0, 0.6,0, -0.05,-0.06,
  ],
  [
 0.25, -0.31, 0.1, 0.45, 0.17, 0.15, -0.4, 0.4, 0.2, 0.4,
    0.4, -0.18,0.32, 0.32, 0.12, 0.32, 0.12, -0.24, 0.15,0.12, -0.53, -0.2,0, 0.26,0.26,
  ],
];
const startFcData = [
  [
      -0.42,
      0.87,
      0.6,
      0.36,
      -0.06,
      0.32,
      -0.59,
      1.15,
      -1.1,
      -0.79,
      -0.96,
      -0.17,
      -0.64,
      0.7,
      -0.2,
      0.48,
      0.64,
      -1.04,
      -0.17,
      -0.27,
      1.16,
      -0.17,
      -0.42,
      -0.54,
      -1.12
  ],
  [
      1.13,
      0.18,
      -0.57,
      0.52,
      0.56,
      1.08,
      -0.53,
      0.16,
      0.7,
      -0.7,
      0.95,
      -0.2,
      0.06,
      -1.1,
      -0.51,
      -1.04,
      1.06,
      0.93,
      -0.58,
      0.23,
      -0.1,
      -0.87,
      0.24,
      -0.48,
      0.26
  ],
  [
      -0.28,
      -1.2,
      -0.03,
      1.17,
      1.14,
      -0.25,
      -0.52,
      -0.86,
      -0.81,
      0.07,
      -0.29,
      -0.39,
      0.9,
      -0.48,
      -0.35,
      0.61,
      -0.08,
      -0.14,
      -0.68,
      -0.6,
      -0.04,
      0.61,
      0.31,
      -0.27,
      0.27
  ],
  [
      -0.66,
      0.71,
      0.91,
      -0.09,
      1.18,
      -0.59,
      -0.71,
      0.11,
      -0.32,
      -0.88,
      0.28,
      0.82,
      -0.93,
      -1.02,
      -0.58,
      -0.95,
      -0.52,
      -0.77,
      0.97,
      -0.49,
      0.92,
      -0.45,
      0.79,
      -0.16,
      0.22
  ],
  [
      0.45,
      1.11,
      -0.03,
      -1.04,
      0.41,
      0.31,
      -0.72,
      1.01,
      1.06,
      0.47,
      -0.44,
      -0.32,
      0.82,
      1.1,
      0.05,
      0.16,
      0.57,
      -0.75,
      -0.93,
      -0.62,
      -0.21,
      -0.28,
      0.04,
      -0.64,
      -0.29
  ]
]
const startFtData =[
  [
      0.72,
      0.07,
      -0.59,
      -0.05,
      1.03,
      -0.94,
      -0.59,
      -0.19,
      0.14,
      -1.05,
      0.69,
      0.96,
      -0.27,
      0.27,
      0.19,
      -0.94,
      0.05,
      0.34,
      0.37,
      0.43,
      0.38,
      0.52,
      1,
      0.11,
      -0.69,
      -0.96,
      1,
      0.47,
      -0.61,
      -0.02,
      -0.28,
      -0.43,
      1.14,
      -0.63,
      -0.41,
      -1.1,
      -0.4,
      -0.62,
      0.53,
      -0.21,
      1.17,
      -0.47,
      0.98,
      0.68,
      0.31,
      0.76,
      0.16,
      0.31,
      0.12,
      0.98,
      0.57,
      0.66,
      0.07,
      0.11,
      -0.7,
      0.18,
      0.4,
      0.44,
      1.04,
      0.49,
      -0.87
  ],
  [
      0.43,
      0.34,
      -0.22,
      -0.99,
      -0.96,
      -0.74,
      0.12,
      1.1,
      -0.24,
      0.17,
      0.56,
      1.15,
      0.13,
      0.9,
      0.43,
      1.01,
      0.96,
      -0.62,
      0.35,
      0.99,
      1.07,
      0.59,
      -0.13,
      0.1,
      -0.84,
      1.19,
      0.22,
      0.75,
      0.15,
      -0.63,
      0.33,
      -0.42,
      0.9,
      -1.11,
      0.55,
      0.14,
      -0.12,
      0.78,
      -0.53,
      0.73,
      -0.89,
      0.46,
      1.17,
      0.66,
      1.11,
      0.07,
      -1.06,
      -0.21,
      0.38,
      -0.61,
      -0.72,
      0.76,
      -1,
      0.7,
      -0.81,
      -0.83,
      0.92,
      0.44,
      -1.18,
      0.24,
      0.26
  ],
  [
      -0.47,
      0.82,
      0.47,
      -0.65,
      0.62,
      1.18,
      0.58,
      -0.08,
      -0.43,
      0.24,
      -1.19,
      0.4,
      -0.05,
      0.19,
      -1.12,
      0.72,
      0.03,
      0.78,
      1.09,
      -0.1,
      0.77,
      1.02,
      0.93,
      1.14,
      -0.9,
      0.23,
      -0.22,
      -0.17,
      0.1,
      0.2,
      1.01,
      -0.2,
      0.23,
      -0.93,
      -0.39,
      1.13,
      0.79,
      -0.74,
      0.33,
      -0.22,
      0.95,
      -1.04,
      0.38,
      0.74,
      -0.8,
      0.03,
      0.31,
      -0.74,
      -0.28,
      -0.84,
      0.05,
      0.74,
      0.7,
      -0.62,
      -0.07,
      0.13,
      -0.06,
      -0.1,
      0.3,
      0.47,
      1.13
  ],
  [
      1.09,
      0.14,
      -0.18,
      0.47,
      0.12,
      0.77,
      0.47,
      0.73,
      1.05,
      -0.91,
      0.16,
      0.43,
      0.97,
      0.9,
      -0.63,
      -0.94,
      -0.53,
      0.22,
      0.98,
      -0.86,
      0.83,
      0.62,
      0.73,
      0.78,
      0.18,
      -1.1,
      0.99,
      0.5,
      0.56,
      0.12,
      -0.36,
      0.87,
      -0.76,
      -0.91,
      -1.19,
      1.1,
      -0.74,
      0.44,
      0.29,
      0.95,
      0.01,
      -0.71,
      1.05,
      0.4,
      -0.97,
      0.51,
      -0.73,
      0.93,
      -0.27,
      1.05,
      -0.16,
      -0.93,
      1.08,
      0.13,
      -0.53,
      -0.04,
      0.03,
      0.02,
      1.03,
      0.59,
      -0.95
  ],
  [
      -0.37,
      -0.81,
      -1.17,
      -0.82,
      1.19,
      -0.3,
      0.19,
      -0.41,
      -1.17,
      1.04,
      -0.07,
      -1.18,
      0.39,
      -0.61,
      -0.62,
      -1.01,
      -0.61,
      -0.72,
      -0.04,
      -0.47,
      -0.65,
      -1.03,
      -0.17,
      -1.11,
      -0.12,
      -0.21,
      0.35,
      -1.19,
      0.86,
      0.43,
      0.24,
      1.18,
      -0.35,
      -0.63,
      -1.2,
      -1.06,
      -0.5,
      1.2,
      -0.57,
      0.05,
      -1.11,
      0.31,
      0.11,
      -0.64,
      -0.81,
      0.38,
      0.82,
      -0.68,
      1.05,
      1.17,
      -1.07,
      0.37,
      -0.21,
      0.34,
      1.2,
      0.74,
      1.09,
      -0.1,
      0.85,
      0.47,
      -0.82
  ]
] 

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