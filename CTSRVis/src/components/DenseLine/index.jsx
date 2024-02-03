import { useEffect, useRef } from "react";
import * as d3 from "d3";
import throttle from "lodash/throttle";

import { getDenseLine, getDensePoint } from "../../apis/api";
import "./index.less";

const COLORGROUP = 6; // 图例分组数

const DenseLine = (props) => {
  const denseRef = useRef(null);
  const { mode, dataset_name } = props;

  useEffect(() => {
    const { width, height } = denseRef.current.getBoundingClientRect();
    getDenseLine('pollutant', dataset_name).then((res) => {
      drawDense(res, dataset_name, width, height);
    });
  }, [denseRef.current, dataset_name]);
  useEffect(() => {}, []);

  const drawDense = (data, dataset_name, w, h) => {
    const timeStamps = Object.keys(data).length;
    const series_length = data[0].length;
    const timeGap = parseInt(timeStamps / COLORGROUP);
    const legendData = [];
    for (let i = 0; i <= COLORGROUP; i++) {
      legendData.push(timeGap * i);
    }

    const flat_data = Object.values(data).flat();
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

    d3.selectAll("div.denseline svg").remove();
    const svg = d3
      .select(".denseline")
      .append("svg")
      .attr("width", dimensions.width)
      .attr("height", dimensions.height)
      .attr("viewBox", [0, 0, dimensions.width, dimensions.height])
      .style("max-width", "100%")
      .style("background", "#fff");
    const bounds = svg
      .append("g")
      .style(
        "transform",
        `translate(${dimensions.margin.left}px, ${
          dimensions.margin.top + dimensions.legendMargin
        }px)`
      );

    // --------------- set scale ---------------
    let xScale = d3
      .scaleBand()
      .domain(d3.range(timeStamps))
      .range([0, boundedWidth]);
    let yScale = d3.scaleBand().domain(d3.range(11)).range([boundedHeight, 0]);
    let colorScale = d3.scaleSequential(
      [timeStamps, -parseInt(timeStamps / 2)],
      d3.interpolateRdYlBu
    );

    // --------------- draw axis -----------------
    let xAxis = d3
      .axisBottom()
      .scale(xScale)
      .tickValues(xScale.domain().filter((d, i) => i % 10 === 0));
    let yAxis = d3
      .axisLeft()
      .scale(yScale)
      .ticks(boundedHeight / 2);
    const axes_x = bounds
      .append("g")
      .style(
        "transform",
        `translate(${xScale.bandwidth() / 2}px, ${
          boundedHeight - dimensions.margin.bottom + yScale.bandwidth() / 2
        }px)`
      );
    const axes_y = bounds
      .append("g")
      .style(
        "transform",
        `translate(${xScale.bandwidth() / 2}px, -${yScale.bandwidth() / 2}px)`
      );
    axes_x.call(xAxis).call((g) => g.select(".domain").remove());
    axes_y.call(yAxis).call((g) => g.select(".domain").remove());

    // --------------- draw gray rect -------------
    const gray_rect = bounds.append("g").attr("class", "dense-gray-rect");
    // --------------- draw tooltip ---------------
    const tooltip = d3
      .select("#dense-tooltip")
      .style("opacity", 0)
      .style("pointer-events", "none");
    // --------------- draw circle ---------------
    const circles = bounds
      .append("g")
      .style("transform", `translate(${dimensions.axisMargin.left}px, 0px)`);
    const bullets = bounds.append("g").attr("class", "dense-bullets");

    const drawBullets = (svgnode, dataset_name, x, y) => {
      getDensePoint(mode, dataset_name, x, y).then((bullets_data) => {
        for (let bts in bullets_data) {
          svgnode
            .append("g")
            .selectAll("circle")
            .data(bullets_data[bts])
            .join("circle")
            .attr("class", "dense-bullets-circle")
            .attr("cx", (d, i) => {
              return xScale.bandwidth() * (parseInt(bts) + 1);
            })
            .attr("cy", (d, i) => {
              return boundedHeight - yScale.bandwidth() * (d + 1);
            })
            .attr("fill", "#eee")
            .attr("r", xScale.bandwidth() * 0.1);
        }
      });
    };

    circles
      .append("g")
      .selectAll("circle")
      .data(flat_data)
      .join("circle")
      .attr("class", "dense-circle")
      .attr("cx", (_, i) => xScale.bandwidth() * parseInt(i / series_length))
      .attr(
        "cy",
        (_, i) =>
          boundedHeight - yScale.bandwidth() * (parseInt(i % series_length) + 1)
      )
      .attr(
        "key",
        (_, i) =>
          parseInt(i / series_length) + "-" + (parseInt(i % series_length) + 1)
      )
      .attr("fill", (d) => colorScale(d))
      .attr("r", xScale.bandwidth() * 0.5)
      .on(
        "mouseover",
        throttle(function (event, d) {
          let key = d3.select(this).attr("key");
          let [x, y] = key.split("-");
          // 0. request data
          drawBullets(bullets, dataset_name, x, y);
          // 1. show the number of each circle
          tooltip.transition().duration(50).style("opacity", 1);
          d3.select("#dense-tooltip")
            .text(d)
            .style(
              "left",
              `${
                xScale.bandwidth() * (parseInt(x) + 0.8) +
                dimensions.margin.left
              }px`
            )
            .style(
              "top",
              `${
                boundedHeight -
                yScale.bandwidth() * (parseInt(y) + 0.3) +
                dimensions.margin.top +
                dimensions.legendMargin
              }px`
            );

          // 2. draw a vertical rect
          gray_rect
            .selectAll("rect")
            .data([d])
            .join("rect")
            .attr(
              "x",
              xScale.bandwidth() * (parseInt(x) - 1) + dimensions.margin.left
            )
            .attr("y", -yScale.bandwidth() / 2)
            .attr("width", xScale.bandwidth())
            .attr("height", boundedHeight)
            .attr("rx", xScale.bandwidth() * 0.2)
            .attr("ry", yScale.bandwidth() * 0.2)
            .attr("fill", "rgba(218, 218, 218, 0.5)");
        }, 2000)
      );
    // --------------- draw legend ---------------
    const legend = bounds
      .append("g")
      .style(
        "transform",
        `translate(${
          boundedWidth -
          dimensions.legendLength * COLORGROUP -
          dimensions.axisMargin.left
        }px, -${dimensions.legendMargin * 2}px)`
      )
      .attr("class", "dense-legend");
    legend
      .selectAll(null)
      .data(legendData)
      .join("rect")
      .attr("x", (_, i) => i * dimensions.legendLength)
      .attr("y", dimensions.legendHeight)
      .attr("width", dimensions.legendLength)
      .attr("height", dimensions.legendHeight)
      .attr("fill", colorScale)
      .on("mouseover", function (event, d) {
        d3.selectAll(".dense-circle")
          .filter((c, ci) => {
            return c < d || c >= d + timeGap;
          })
          .style("opacity", 0.3);
      })
      .on("mouseout", function (event, d) {
        d3.selectAll(".dense-circle").style("opacity", 1);
      });

    legend
      .selectAll(null)
      .data(legendData)
      .join("text")
      .attr("class", "dense-legend-text")
      .text((d, i) => {
        if (i === COLORGROUP) {
          return ">=" + d;
        } else return d;
      })
      .attr("x", (_, i) => i * dimensions.legendLength)
      .attr("y", dimensions.legendHeight * 3.6);
  };
  return (
    <div
      className="denseline"
      ref={denseRef}
      onMouseOut={() => {
        d3.selectAll(".dense-bullets-circle").remove();
      }}
    >
      <div id="dense-tooltip"></div>
    </div>
  );
};

export default DenseLine;
