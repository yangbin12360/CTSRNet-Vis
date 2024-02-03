import { useEffect, useRef, useState } from "react";
import * as d3 from "d3";

import "./index.css";
import { getContrastHeat } from "../../apis/api";
import { SENSORLIST_NAME_EN1 } from "../../utils/constants";
import { range } from "lodash";

const ContrastHeat = (props) => {
  const {
    timeStamp
  } = props;
  const contrastHeatRef = useRef(null);
  //自定义颜色插值器
  const customInterpolator = (t) => {
    const startColor = d3.rgb("#84d59d"); //负  
    const endColor = d3.rgb("#f6b771"); //正 
    
    return d3.rgb(
      startColor.r + t * (endColor.r - startColor.r) ,
      startColor.g + t * (endColor.g - startColor.g),
      startColor.b + t * (endColor.b - startColor.b)
    );
  };
  // 自定义颜色插值器
// function customColorInterpolator(value) {
//   // 定义颜色比例尺
//   const colorScale = d3.scaleLinear()
//     .domain([-1,  1])  // 输入值的范围
//     .range(['green', 'red'])  // 对应的颜色值
//     .interpolate(d3.interpolateRgb);  // 使用RGB颜色插值

//   return colorScale(value);
// }
  // const customColorInterpolator = (value)=>{
  //   if (value < 0) {
  //     // 蓝色渐变，从浅蓝色到深蓝色
  //     return d3.interpolateBlues(-value);
  //   } else if (value > 0) {
  //     // 红色渐变，从浅红色到深红色
  //     return d3.interpolateReds(value);
  //   } else {
  //     // 当值等于0时，使用白色
  //     return "#6CE68E";
  //   }
  // }
  //计算同一站点不同时间的污染物浓度差百分比（含负数）
  const contrastValue = (pollutantIndex, sensorIndex, baseData, targetData) => {
    // if(timeIndex != 0 ) timeIndex= timeIndex-1;
    let contrastData = baseData[sensorIndex];
    return parseFloat(targetData["data"] - contrastData) / contrastData;
  };

  // 绘制对比热力图
  const drawContrastHeat = (data, label, times, w, h) => {
    //污染物标签和时间戳数据
    const pollutantLabel = label;
    const timeData = times;
    let sensorData = data;
    console.log("sensorData", sensorData);
    //中间时间戳的数据   25x10
    // let baseData = sensorData[parseInt(sensorData.length / 2)];
    //创建画布
    let dimensions = {
      width: w,
      height: h,
      margin: {
        top: 30,
        right: 30,
        bottom: 30,
        left: 30,
      },
      axisMargin: {
        left: 20,
        bottom: 10,
      },
      legendMargin: 20,
      legendLength: 40,
      legendHeight: 8,
    };
    let boundedWidth =
      dimensions.width - dimensions.margin.left - dimensions.margin.right;
    let boundedHeight =
      dimensions.height - dimensions.margin.top - dimensions.margin.bottom;
    d3.selectAll("div.contrastheat svg").remove();
    const svg = d3
      .select(".contrastheat")
      .append("svg")
      .attr("width", dimensions.width)
      .attr("height", dimensions.height)
      .attr("viewbox", [0, 0, dimensions.width, dimensions.height])
      .style("max-width", "100%")
      .style("background", "#f7f7f7");
    const bounds = svg
      .append("g")
      .style(
        "transform",
        `translate(${dimensions.margin.left}px, ${dimensions.margin.top}px)`
      );
    //创建X轴和Y轴的比例尺
    let xScale = d3
      .scaleBand()
      .domain(timeData.map((d) => d))
      .range([0, boundedWidth])
      .padding(0.1);
    let yScale = d3
      .scaleBand()
      .domain(pollutantLabel.map((d) => d))
      .range([0, boundedHeight]);
    //创建站点颜色比例尺
    const colorList = [
      "#001219",
      "#005f73",
      "#0a9396",
      "#94d2bd",
      "#e9d8a6",
      "#ee9b00",
      "#ca6702",
      "#bb3e03",
      "#ae2012",
      "#9b2226",
    ];
    let sensorColorSacle = d3
      .scaleOrdinal()
      .domain(range(0, 10))
      .range(colorList);
    //创建颜色相关性比例尺
    const correlationColorScale = d3
      .scaleSequential()
      .domain([-1, 1])
      .interpolator(customInterpolator);
    //创建X轴和Y轴
    let xAxis = d3.axisTop(xScale);
    let yAxis = d3.axisLeft(yScale);
    const axes_x = bounds
      .append("g")
      .attr("class", "x-axis")
      .style("transform", `translate(${xScale.bandwidth() / 3}px, ${0}px)`);
    const axes_y = bounds
      .append("g")
      .attr("class", "y-axis")
      .style(
        "transform",
        `translate(${yScale.bandwidth() / 3 + 10}px, ${10}px)`
      );
    axes_x
      .call(xAxis)
      .call((g) => g.select(".domain").remove())
      .selectAll(".tick line")
      .remove();
    // 修改x轴的字体类型和字体大小
    axes_x
      .selectAll(".tick text")
      .style("font-family", "Arial") 
      .style("font-size", "15px")
      .style("font-weight","bold")

    axes_y
      .call(yAxis)
      .call((g) => g.select(".domain").remove())
      .selectAll(".tick line")
      .remove();
    // 修改y轴的字体类型和字体大小
    axes_y
      .selectAll(".tick text")
      .style("font-family", "Arial") 
      .style("font-size", "15px")
      ; 
      let middleColumnIndex = Math.floor(sensorData[0].length / 2);
    // 全局变量用于跟踪当前选择的扇形
    let selectedSlice = null;
    //绘制十等分的饼图在对应xy轴上
    let pie = d3
      .pie()
      .value((d) => d)
      .sort(null);
    // 创建提示框元素
    const tooltip = d3
      .select("body")
      .append("div")
      .attr("class", "tooltip")
      .style("opacity", 0)
      .style("position", "absolute")
      .style("background-color", "white")
      .style("border", "1px solid black")
      .style("border-radius", "5px")
      .style("padding", "5px");
    //循环绘制饼图
    sensorData.forEach((rowArray, rowIndex) => {
      //一次一行 // rowIndex 0  =>  24
      rowArray.forEach((columnElement, columnIndex) => {
        // coLoumnIndex 0  => 6    

        if (columnIndex === middleColumnIndex) {
          // 左边框
          bounds
            .append("line")
            .attr("x1", xScale(timeData[columnIndex]) + (5 * xScale.bandwidth()) / 6 - 25)
            .attr("y1", yScale.bandwidth() / 2 + 10 + yScale(pollutantLabel[rowIndex]) - 21)
            .attr("x2", xScale(timeData[columnIndex]) + (5 * xScale.bandwidth()) / 6 - 25)
            .attr("y2", yScale.bandwidth() / 2 + 10 + yScale(pollutantLabel[rowIndex]) + 23)
            .attr("stroke", "black")
            .attr("stroke-width", 2);
        
          // 右边框
          bounds
            .append("line")
            .attr("x1", xScale(timeData[columnIndex]) + (5 * xScale.bandwidth()) / 6 + 25)
            .attr("y1", yScale.bandwidth() / 2 + 10 + yScale(pollutantLabel[rowIndex]) - 21)
            .attr("x2", xScale(timeData[columnIndex]) + (5 * xScale.bandwidth()) / 6 + 25)
            .attr("y2", yScale.bandwidth() / 2 + 10 + yScale(pollutantLabel[rowIndex]) + 23)
            .attr("stroke", "black")
            .attr("stroke-width", 2);
        
          // 如果是第一个元素，添加上边框
          if (rowIndex === 0) {
            bounds
              .append("line")
              .attr("x1", xScale(timeData[columnIndex]) + (5 * xScale.bandwidth()) / 6 - 25)
              .attr("y1", yScale.bandwidth() / 2 + 10 + yScale(pollutantLabel[rowIndex]) - 23)
              .attr("x2", xScale(timeData[columnIndex]) + (5 * xScale.bandwidth()) / 6 + 25)
              .attr("y2", yScale.bandwidth() / 2 + 10 + yScale(pollutantLabel[rowIndex]) - 23)
              .attr("stroke", "black")
              .attr("stroke-width", 2);
          }
        
          // 如果是最后一个元素，添加下边框
          if (rowIndex === sensorData.length - 1) {
            bounds
              .append("line")
              .attr("x1", xScale(timeData[columnIndex]) + (5 * xScale.bandwidth()) / 6 - 25)
              .attr("y1", yScale.bandwidth() / 2 + 10 + yScale(pollutantLabel[rowIndex]) + 23)
              .attr("x2", xScale(timeData[columnIndex]) + (5 * xScale.bandwidth()) / 6 + 25)
              .attr("y2", yScale.bandwidth() / 2 + 10 + yScale(pollutantLabel[rowIndex]) + 23)
              .attr("stroke", "black")
              .attr("stroke-width", 2);
          }}
        // 为每个饼图创建一个分组
        let baseData = rowArray[parseInt(rowArray.length / 2)];
        let pieGroup = bounds
          .append("g")
          .attr("class", "pie-group")
          .style(
            "transform",
            `translate(${
              xScale(timeData[columnIndex]) + (5 * xScale.bandwidth()) / 6
            }px, ${
              yScale.bandwidth() / 2 + 10 + yScale(pollutantLabel[rowIndex])
            }px)`
          );
        // let arcGroup = bounds
        //   .append("g")
        //   .attr("class", "arc-group")
        //   .style(
        //     "transform",
        //     `translate(${
        //       xScale(timeData[columnIndex]) + (5 * xScale.bandwidth()) / 6
        //     }px, ${
        //       yScale.bandwidth() / 2 + 10 + yScale(pollutantLabel[rowIndex])
        //     }px)`
        //   );
        //站点圆弧路径元素添加
        // arcGroup
        //   .selectAll("path")
        //   .data(pie(columnElement))
        //   .enter()
        //   .append("path")
        //   .attr("d", d3.arc().innerRadius(20).outerRadius(25))
        //   .attr("stroke", "white")
        //   .attr("stroke-width", 3)
        //   .style("text", (d, i) => {
        //     return i;
        //   })
        //   .attr("fill", (d, i) => {
        //     return sensorColorSacle(i);
        //   })
        //   // 添加鼠标悬停事件处理程序
        //   .on("mouseover",  (event, d) => {
        //     console.log(d);
        //     tooltip
        //       .style("opacity", 1)
        //       .html(SENSORLIST_NAME_SIM[d["index"]]) // 根据需要设置显示的值
        //       .style("left", event.pageX + 10 + "px")
        //       .style("top", event.pageY - 20 + "px");
        //   })
        //   .on("mousemove", (event, d) => {
        //     tooltip
        //       .style("left", event.pageX + 10 + "px")
        //       .style("top", event.pageY - 20 + "px");
        //   })
        //   .on("mouseout",  () => {
        //     tooltip.style("opacity", 0);
        //   });
        //将路径元素附加到相应的分组上
        pieGroup
          .selectAll("path")
          .data(pie(columnElement))
          .enter()
          .append("path")
          .attr("d", d3.arc().innerRadius(0).outerRadius(19))
          .attr("stroke", "white")
          .attr("stroke-width",1.5)
          .attr("fill", (d, i) => {
            return correlationColorScale(
              contrastValue(columnIndex, i, baseData, d)
            );
          })
          // 添加鼠标悬停事件处理程序
          .on("mouseover", (event, d) => {
            tooltip
              .style("opacity", 1)
              .html(SENSORLIST_NAME_EN1[d["index"]]+':'+d.value) // 根据需要设置显示的值
              .style("left", event.pageX + 10 + "px")
              .style("top", event.pageY - 20 + "px");
          })
          .on("mousemove", (event, d) => {
            tooltip
              .style("left", event.pageX + 10 + "px")
              .style("top", event.pageY - 20 + "px");
          })
          .on("mouseout", () => {
            tooltip.style("opacity", 0);
          })
          .on("click", (event, d, i) => {
            d3.selectAll(".pie-group path").attr("opacity", (d1) => {
              if (d.index != d1.index)
                return this === event.currentTarget && selectedSlice !== this
                  ? 1
                  : 0.1;
            });
            d3.selectAll(".arc-group path").attr("opacity", (d1) => {
              if (d.index != d1.index)
                return this === event.currentTarget && selectedSlice !== this
                  ? 1
                  : 0.1;
            });
            // 检查是否已选择扇形，如果已选择，则重置选择并恢复透明度
            if (selectedSlice === event.currentTarget) {
              selectedSlice = null;
              d3.selectAll(".pie-group path").attr("opacity", 1);
              d3.selectAll(".arc-group path").attr("opacity", 1);
            } else {
              selectedSlice = event.currentTarget;
            }
          });
      });
    });
  };
  const [timeRange, setTimeRange] = useState(3);
  useEffect(() => {
    const { width, height } = contrastHeatRef.current.getBoundingClientRect();
    console.log("timeStamp",timeStamp);
    getContrastHeat(timeStamp, timeRange).then((res) => {
      const { data, label, times } = res;
      drawContrastHeat(data, label, times, width, height);
    });
  }, [timeStamp, timeRange]);

  return <div className="contrastheat" ref={contrastHeatRef}></div>;
};

export default ContrastHeat;
