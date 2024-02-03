import { useEffect, useRef } from "react";
import * as d3 from "d3";

import "./index.less";
import { getAvgLine } from "../../apis/api";
import { COLORLIST } from "../../utils/constants";

const AvgLine = (props) => {
  const {
    mode,
    dataset,
    isAvg,
    feature,
    cluster,
    numOfCluster,
    onTimeStampChange,
  } = props;
  const avgLineRef = useRef(null);

  const drawLines = (data, data_line, label, w, h) => {
    const flat_data = data.flat();
    const label_set = [...new Set(label)];

    const dimensions = {
      width: w,
      height: h,
      margin: {
        top: 20,
        right: 30,
        bottom: 20,
        left: 30,
      },
      legendmargin: 40,
      legendlength: 16,
    };

    const boundedWidth =
      dimensions.width - dimensions.margin.left - dimensions.margin.right;
    const boundedHeight =
      dimensions.height -
      dimensions.margin.top -
      dimensions.margin.bottom -
      dimensions.legendmargin;

    d3.selectAll("div.avgline svg").remove();
    const svg = d3
      .select(".avgline")
      .append("svg")
      .attr("width", dimensions.width)
      .attr("height", dimensions.height)
      .attr("viewbox", [0, 0, dimensions.width, dimensions.height])
      .style("max-width", "100%")
      .style("background", "#fff");
    const bounds = svg
      .append("g")
      .style(
        "transform",
        `translate(${dimensions.margin.left}px, ${dimensions.margin.top}px)`
      );

    let xScale = d3
      .scaleLinear()
      .domain([0, data[0].length])
      .range([0, boundedWidth]);
    let yScale = d3
      .scaleLinear()
      .domain(d3.extent(flat_data))
      .range([boundedHeight, 0]);
    let colorScale = d3.scaleOrdinal().domain(label_set).range(COLORLIST);

    let xAxis = d3.axisBottom().scale(xScale).ticks(5);
    let yAxis = d3.axisLeft().scale(yScale).ticks(8);

    const hover_line = bounds
      .append("line")
      .attr("class", "line-axis-highlight")
      .attr("opacity", 0);

    const hover_text = d3.select("div.x_axis_tooltip").style("opacity", 0);

    const line_g = bounds
      .append("g")
      .attr("class", "line-g")
      .on("mousemove", function (event) {
        let x = d3.pointer(event, this)[0];
        let y = d3.pointer(event, this)[1];
        let time_stamp = Math.round(xScale.invert(d3.pointer(event, this)[0])); // 获取当前时间戳
        let values_at_stamp = data.map((item) => item[parseInt(time_stamp)]); // 获取当前时间戳对应的数据

        hover_line
          .attr("x1", x)
          .attr("x2", x)
          .attr("y1", boundedHeight)
          .attr("y2", 0)
          .attr("stroke", "#666")
          .attr("stroke-width", 2)
          .attr("stroke-dasharray", "8,2")
          .attr("opacity", 1);

        hover_text.transition().duration(50).style("opacity", 1);
        let tex = catAxisToolTipText(values_at_stamp, label);
        d3.select("div.x_axis_tooltip")
          .html(tex)
          .style("top", parseInt(y) + "px")
          .style("left", parseInt(x + 40) + "px");
      })
      .on("mouseout", function (event) {
        hover_text.transition().duration(50).style("opacity", 0);
      })
      .on("click", function (event, d) {
        let time_stamp = parseInt(xScale.invert(d3.pointer(event, this)[0]));
        onTimeStampChange(time_stamp);
      });

    // --------------- draw line ---------------
    line_g
      .selectAll("path")
      .data(data_line)
      .join("path")
      .attr("class", "line")
      .attr("key", (_, i) => label[i])
      .attr("d", (d, i) => {
        return d3
          .line()
          .x((d) => xScale(d.index))
          .y((d) => yScale(d.value))(d);
      })
      .attr("fill", "none")
      .attr("stroke", (d, i) => colorScale(label[i]))
      .attr("stroke-width", 1);

    // --------------- draw axis ---------------
    bounds
      .append("g")
      .call(xAxis)
      .style("transform", `translateY(${boundedHeight}px)`)
      .call((g) => g.select(".domain").remove())
      .call((g) =>
        g
          .selectAll(".tick line")
          .clone()
          .attr("y2", -(boundedHeight + dimensions.margin.top * 0.5))
          .style("opacity", 0.1)
      );
    bounds
      .append("g")
      .call(yAxis)
      .call((g) => g.select(".domain").remove())
      .call((g) =>
        g
          .selectAll(".tick line")
          .clone()
          .attr("x2", boundedWidth + dimensions.margin.left * 0.5)
          .style("opacity", 0.1)
      );

    // --------------- draw legend ---------------
    const legend = bounds
      .append("g")
      .style(
        "transform",
        `translateY(${boundedHeight + dimensions.legendmargin * 0.8}px)`
      );
    legend
      .selectAll(null)
      .data(label_set)
      .join("rect")
      .attr("x", (d, i) => parseInt(boundedWidth / label_set.length) * i)
      .attr("width", dimensions.legendlength)
      .attr("height", dimensions.legendlength)
      .attr("fill", (d) => colorScale(d))
      .on("mouseover", function (event, d) {
        // 将当前类别加粗
        d3.selectAll(".line")
          .filter((l, li) => {
            if (isAvg) {
              return label_set[li] === d;
            } else {
              return label_set[parseInt(li / 5)] === d;
            }
          })
          .attr("stroke-width", 2);
        // 将未选中的类别的元素改为灰色
        d3.selectAll(".line")
          .filter((l, li) => {
            if (isAvg) {
              return label_set[li] !== d;
            } else {
              return label_set[parseInt(li / 5)] !== d;
            }
          })
          .attr("stroke", "#e3e3e3");
        // 高亮散点图中数据
        d3.selectAll(".scatter .scatter-circle")
          .filter((c, ci) => {
            return c[3] === d;
          })
          .attr("r", 6);
      })
      .on("mouseout", function (event) {
        //  恢复所有元素颜色
        d3.selectAll(".line")
          .attr("stroke", (d, i) => colorScale(label[i]))
          .attr("stroke-width", 1);
        d3.selectAll(".scatter .scatter-circle").attr("r", 4);
      });
    legend
      .selectAll(null)
      .data(label_set)
      .join("text")
      .attr("class", "legendtext")
      .text((d) => d)
      .attr(
        "x",
        (d, i) =>
          parseInt(boundedWidth / label_set.length) * i +
          dimensions.legendlength * 1.5
      )
      .attr("y", dimensions.legendlength * 0.8)
      .style("font-family", "sans-serif")
      .style("font-size", `${dimensions.legendlength * 0.8}px`);
  };

  const catAxisToolTipText = (values, labels) => {
    let cat_values = values.map((item, index) => {
      return labels[index] + ": " + item.toFixed(2) + "\n";
    });
    return [...cat_values].join("");
  };

  useEffect(() => {
    const { width, height } = avgLineRef.current.getBoundingClientRect();

    getAvgLine(mode, dataset, isAvg, feature, cluster, numOfCluster).then(
      (res) => {
        let { data, label } = res;
        let data_line = [];
        for (let i = 0; i < data.length; i++) {
          let line = data[i].map((d, i) => {
            return {
              index: i.toString(),
              value: d.toString(),
            };
          });
          data_line.push(line);
        }

        // data, label 为请求返回的原始数据
        // data_line 为用于绘制折线图的处理后数据, 数据中每个时间点处理为对象
        drawLines(data, data_line, label, width, height);
      }
    );
  }, [
    avgLineRef.current,
    mode,
    dataset,
    isAvg,
    feature,
    cluster,
    numOfCluster,
  ]);
  return (
    <div className="avgline" ref={avgLineRef}>
      <div className="x_axis_tooltip"></div>
    </div>
  );
};

export default AvgLine;
