import React, { useEffect, useRef, useState } from "react";
import { Radio, Button, Select, Form } from "antd";
import * as d3 from "d3";
import { lasso } from "d3-lasso";

import { getCluster } from "../../apis/api";
import { COLORLIST, SENSORLIST_NAME, SENSORLIST, SENSORLIST_NAME_EN1 } from "../../utils/constants";

import "./index.less";

const ClusterScatter = (props) => {
  const {
    dataset,
    cluster,
    numOfCluster,
    embeddingMethod,
    dimReduction,
    setClusterLabel,
    setChangedLabel,
  } = props;

  const clusterDataRef = useRef(null);
  const originLabelDataRef = useRef(null);
  const clusterLabelDataRef = useRef(null);
  const newLabelDataRef = useRef(null);

  const [colorEncode, setColorEncode] = useState("cluster");
  const [labelOptions, setLabelOptions] = useState([]);

  const clusterRef = useRef(null);

  const [form] = Form.useForm();

  const handleColorEncodingChange = (e) => {
    setColorEncode(e.target.value);
  };

  const onNewLabelChange = (f) => {
    newLabelDataRef.current = f;
  };

  const onResetClick = () => {
    // 恢复 FC 图
    // getFeatureContribution(
    //   dataset,
    //   featureTimestamp,
    //   clusterLabelDataRef.current
    // ).then((res) => {
    //   let { fc } = res;
    //   handleFCDataChange(fc);
    // });
    // 恢复 Scatter
    const { width, height } = clusterRef.current.getBoundingClientRect();
    drawScatter(clusterDataRef.current, width, height);
    // 恢复 Cluster Page
    setChangedLabel([]);
  };

  const changeLabel = (data) => {
    // 将选中的部分数据修改为新的label
    // 根据选中的数据, 选择的label, 修改画图的cluster数据
    let idxes = data.map((item) => {
      return item[0];
    }); // 被选中的数据在原始聚类数据中的索引
    let changedClusterData;
    if (colorEncode === "cluster") {
      changedClusterData = clusterDataRef.current.map((item, index) => {
        if (idxes.indexOf(index) !== -1) {
    
          return [...item.slice(0, 3), newLabelDataRef.current.toString()];
        } else {
          return item;
        }
      });
      clusterLabelDataRef.current = changedClusterData.map((item) => item[3]);
      setClusterLabel(clusterLabelDataRef.current);
    } else {
      changedClusterData = clusterDataRef.current.map((item, index) => {
        if (idxes.indexOf(index) !== -1) {
          return [
            ...item.slice(0, 2),
            newLabelDataRef.current.toString(),
            ...item.slice(-1),
          ];
        } else {
          return item;
        }
      });
      clusterLabelDataRef.current = changedClusterData.map((item) => item[2]);
      setClusterLabel(clusterLabelDataRef.current);
    }
    // getFeatureContribution(
    //   dataset,
    //   featureTimestamp,
    //   clusterLabelDataRef.current
    // ).then((res) => {
    //   let { fc } = res;
    //   handleFCDataChange(fc);
    // });
    setChangedLabel(clusterLabelDataRef.current);

    const { width, height } = clusterRef.current.getBoundingClientRect();
    drawScatter(changedClusterData, width, height);
  };

  const drawScatter = (data, w, h) => {
    let dimensions = {
      width: w,
      height: h,
      margin: {
        top: 30,
        right: 30,
        bottom: 30,
        left: 30,
      },
    };
    let boundedWidth =
      dimensions.width - dimensions.margin.left - dimensions.margin.right;
    let boundedHeight =
      dimensions.height - dimensions.margin.top - dimensions.margin.bottom;
    d3.selectAll("div.scatter svg").remove();
    const svg = d3
      .select(".scatter")
      .append("svg")
      .attr("width", dimensions.width)
      .attr("height", dimensions.height)
      .attr("viewBox", [0, 0, dimensions.width, dimensions.height])
      .style("max-width", "100%")
      .style("background", "#fff")
      .on("click", function (event) {
        d3.selectAll(".selected").classed("selected", false);
      });

    const bounds = svg
      .append("g")
      .style(
        "transform",
        `translate(${dimensions.margin.left}px, ${dimensions.margin.top}px)`
      );

    const xAccessor = (d) => parseFloat(d[0]); // x
    const yAccessor = (d) => parseFloat(d[1]); // y
    const clusterLabelMax = d3.max(clusterLabelDataRef.current.map(parseInt));
    const clusterLabelDomain = Array.from(Array(clusterLabelMax).keys());

    let [xmin, xmax] = d3.extent(data, xAccessor);
    let [ymin, ymax] = d3.extent(data, yAccessor);
    let xdiff = (xmax - xmin) / 10;
    let ydiff = (ymax - ymin) / 10;
    let xScale = d3
      .scaleLinear()
      .domain([xmin - xdiff, xmax + xdiff])
      .range([
        dimensions.margin.left,
        dimensions.width - dimensions.margin.left,
      ])
      .nice();
    let yScale = d3
      .scaleLinear()
      .domain([ymin - ydiff, ymax + ydiff])
      .range([dimensions.height - dimensions.margin.top, dimensions.margin.top])
      .nice();
    let colorScale = d3
      .scaleOrdinal()
      .domain(colorEncode === "cluster" ? clusterLabelDomain : SENSORLIST_NAME)
      .range(COLORLIST);
    let colorScaleYear = d3
      .scaleLinear()
      .domain([0, 1, 2, 3, 4])
      .range(["#bfcde5", "#abacd6", "#67c07c", "#ffae51", "#ff7f53"]);
    // let radius = Math.floor(boundedWidth * 0.01);
    let radius = 5;

    let xAxis = (g, scale) =>
      g
        .attr("transform", `translate(0,${yScale(ymin - ydiff)})`)
        .call(d3.axisBottom(scale).ticks(8));
    let yAxis = (g, scale) =>
      g
        .attr("transform", `translate(${xScale(xmin - xdiff)},0)`)
        .call(d3.axisLeft(scale).ticks(8));
    const tooltipDiv = d3
      .select("#scatter-tooltip")
      .style("opacity", 0)
      .style("pointer-events", "none");

    const circles = bounds
      .selectAll("circle")
      .data(data)
      .join("circle")
      .attr("class", "scatter-circle")
      .attr("cx", (d) => xScale(xAccessor(d)))
      .attr("cy", (d) => yScale(yAccessor(d)))
      .attr("fill", (d, i) => {
        console.log("现在测试:",d[2]);
        return String(
          colorScale(colorEncode === "cluster" ? parseInt(d[3]) : d[2].split('_')[0])
        );
      })
      .attr("stroke", (d, i) => {
        if (colorEncode === "sensor")
          return colorScaleYear(d[2].split("_")[1].slice(1));
        else return "rgba(0, 0, 0, 0)";
      })
      .attr("stroke-width", 2)
      .attr("r", radius)
      .attr("id", (d, i) => i)
      .on("mouseover", function (event, d, i) {
        d3.select(this)
          .transition()
          .duration(50)
          .attr("r", radius * 1.5);
        tooltipDiv.transition().duration(50).style("opacity", 1);
        const name = d[2].substring(0,d[2].indexOf("_"));
        const enName = SENSORLIST_NAME_EN1[SENSORLIST_NAME.indexOf(name)]+ d[2].substring(d[2].indexOf("_"),d[2].length);
        d3.select("#scatter-tooltip")
          .html(enName  + "\n Cluster " + d[3])
          .style("left", `${event.offsetX + 10}px`)
          .style("top", `${event.offsetY - 15}px`);
      })
      .on("mouseout", function () {
        d3.select(this).transition().duration(50).attr("r", radius);
        tooltipDiv.transition().duration(50).style("opacity", 0);
      });

    // ----------------   LASSO STUFF . ----------------
    const lasso_start = () => {
      my_lasso
        .items()
        // .attr("r", 7)
        .classed("not_possible", true)
        .classed("selected", false);
    };

    const lasso_draw = () => {
      my_lasso
        .possibleItems()
        .classed("not_possible", false)
        .classed("possible", true);
      my_lasso
        .notPossibleItems()
        .classed("not_possible", true)
        .classed("possible", false);
    };

    const lasso_end = () => {
      my_lasso
        .items()
        .classed("not_possible", false)
        .classed("possible", false);
      my_lasso.selectedItems().classed("selected", true);
      let selectedData = my_lasso.selectedItems()._groups[0].map((item) => {
        return [parseInt(item.getAttribute("id")), item.__data__];
      });
      changeLabel(selectedData);
      // console.log("selected: ", selectedData);
      // .attr("r", 7);
    };

    const my_lasso = lasso()
      .closePathDistance(305)
      .closePathSelect(true)
      .targetArea(bounds)
      .items(circles)
      .on("start", lasso_start)
      .on("draw", lasso_draw)
      .on("end", lasso_end);

    bounds.call(my_lasso);
    // ----------------   LASSO STUFF END. ----------------
    // ----------------------zoom相关 Start----------------------------
    const gx = svg.append("g").attr("id", "xAxis");
    const gy = svg.append("g").attr("id", "yAxis");
    let z = d3.zoomIdentity;
    // set up the ancillary zooms and an accessor for their transforms
    const zoomX = d3.zoom().scaleExtent([0.1, 10]);
    const zoomY = d3.zoom().scaleExtent([0.2, 5]);
    const tx = () => d3.zoomTransform(gx.node());
    const ty = () => d3.zoomTransform(gy.node());
    gx.call(zoomX).attr("pointer-events", "none");
    gy.call(zoomY).attr("pointer-events", "none");
    // active zooming
    const zoom = d3.zoom().on("zoom", function (e) {
      const t = e.transform;
      const k = t.k / z.k;
      const point = center(e, this);

      // is it on an axis? is the shift key pressed?
      const doX = point[0] > xScale.range()[0];
      const doY = point[1] < yScale.range()[0];
      const shift = e.sourceEvent && e.sourceEvent.shiftKey;

      if (k === 1) {
        // pure translation?
        doX && gx.call(zoomX.translateBy, (t.x - z.x) / tx().k, 0);
        doY && gy.call(zoomY.translateBy, 0, (t.y - z.y) / ty().k);
      } else {
        // if not, we're zooming on a fixed point
        doX && gx.call(zoomX.scaleBy, shift ? 1 / k : k, point);
        doY && gy.call(zoomY.scaleBy, k, point);
      }
      z = t;
      redraw();
    });
    function redraw() {
      const xr = tx().rescaleX(xScale);
      const yr = ty().rescaleY(yScale);
      // 缩放坐标轴
      gx.call(xAxis, xr);
      gy.call(yAxis, yr);
      // 缩放节点
      circles
        .attr("cx", (d) => xr(d[0]))
        .attr("cy", (d) => yr(d[1]))
        .attr("rx", 6 * Math.sqrt(tx().k))
        .attr("ry", 6 * Math.sqrt(ty().k));
    }
    // center the action (handles multitouch)
    function center(event, target) {
      if (event.sourceEvent) {
        const p = d3.pointers(event, target);
        return [d3.mean(p, (d) => d[0]), d3.mean(p, (d) => d[1])];
      }
      return [dimensions.width / 2, dimensions.height / 2];
    }

    svg.call(zoom).call(zoom.transform, d3.zoomIdentity.scale(0.8));
    //----------------------zoom相关 End ----------------------------
  };

  useEffect(() => {
    getCluster(
      cluster,
      numOfCluster + 1,
      embeddingMethod,
      dataset,
      dimReduction
    ).then((res) => {
      if (clusterRef.current) {
        const { width, height } = clusterRef.current.getBoundingClientRect();
        const { data, origin_label, cluster_label } = res;

        let labelOps = [];
        if (colorEncode === "cluster") {
          for (let i = 0; i <= [...new Set(cluster_label)].length; i++) {
            labelOps.push({ value: i, label: i });
          }
        } else {
          labelOps = SENSORLIST;
        }

        clusterDataRef.current = data;
        originLabelDataRef.current = origin_label;
        clusterLabelDataRef.current = cluster_label;
        setClusterLabel(clusterLabelDataRef.current);
        setLabelOptions(labelOps);

        // getFeatureContribution(
        //   dataset,
        //   featureTimestamp,
        //   clusterLabelDataRef.current
        // ).then((res) => {
        //   let { fc, ft } = res;
        //   handleFTDataChange(ft);
        //   handleFCDataChange(fc);
        // });
        setChangedLabel([]);

        drawScatter(data, width, height);
      }
    });
  }, [
    dataset,
    cluster,
    numOfCluster,
    embeddingMethod,
    dimReduction,
    colorEncode,
    // featureTimestamp,
  ]);

  // useEffect(() => {
  //   console.log("t", featureTimestamp);
  // }, [featureTimestamp]);

  return (
    <div className="scatter-wrap">
      <div className="scatter-control">
        <div className="scatter-control-color">
          <div className="scatter-control-color-text">Color Encode</div>
          <div className="scatter-control-color-radio">
            <Radio.Group
              defaultValue="cluster"
              onChange={handleColorEncodingChange}
              size="small"
            >
              <Radio value={"sensor"}>Sensor</Radio>
              <Radio value={"cluster"}>Cluster</Radio>
            </Radio.Group>
          </div>
        </div>
        <div className="scatter-control-label">
          <div className="scatter-control-label-text">Set as Cluster </div>
          <div className="scatter-control-label-select">
            <Form form={form} layout="inline" size="small">
              <Form.Item>
                <Select
                  defaultValue={
                    colorEncode === "cluster" ? "3" : SENSORLIST[0].value
                  }
                  onChange={onNewLabelChange}
                  options={labelOptions}
                  style={{ width: 100 }}
                ></Select>
              </Form.Item>
              <Form.Item>
                <Button onClick={onResetClick}>reset</Button>
              </Form.Item>
            </Form>
          </div>
        </div>
      </div>
      <div className="scatter" ref={clusterRef}>
        <div id="scatter-tooltip"></div>
      </div>
    </div>
  );
};

export default ClusterScatter;
