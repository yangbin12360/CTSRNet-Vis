import * as d3 from "d3";
import { useEffect, useRef } from "react";
import * as echarts from "echarts";
import * as turf from "@turf/turf";

import { getHKGeoJson } from "../../apis/api";
import {
  SENSORLIST_NAME,
  SENSORLIST_NAME_SIM,
  SENSORLIST_NAME_EN1,
  COLORLIST,
} from "../../utils/constants";
import "./index.less";

const Map = (props) => {
  const { rangeBarTimeStampIdx, rangeBarList, rangeBarPollutantIdx } = props;
  let senIdx = useRef(-1); // punch 与 map 交互时,需要高亮的监测站索引
  const mapRef = useRef(null);
  const punchRef = useRef(null);

  useEffect(() => {
    console.log("rangeBarList,",rangeBarList);
    const { width, height } = mapRef.current.getBoundingClientRect();
    getHKGeoJson(rangeBarPollutantIdx, rangeBarTimeStampIdx, rangeBarList).then(
      (res) => {
        let { hk, sensor, data, punch, punch_max } = res;

        let data_format = {};
        data.forEach((item, index) => {
          data_format[SENSORLIST_NAME[index]] = item;
        });

        punch = punch.map((item, index) => {
          return item.map((it, id) => {
            return [id, index, it];
          });
        });
        punch = punch.flat();

        drawMap(hk, sensor, data_format, width, height);
        drawPunch(punch, punch_max);
      }
    );
  }, [rangeBarTimeStampIdx, rangeBarList, rangeBarPollutantIdx]);

  const zoomed = (e) => {
    const { transform } = e;
    d3.selectAll("div.map svg").attr("transform", transform);
  };

  const drawMap = (geojson, sensors, data, w, h) => {
    // 地图投影
    const projection = d3
      .geoMercator()
      .center([114, 22])
      .scale(h * 220)
      .translate([w / 3.8, h * 2]);
    //  zoom
    const zoom = d3.zoom().scaleExtent([-2, 10]).on("zoom", zoomed);
    // 坐标转换
    const myturf = (feature) => turf.rewind(feature, { reverse: true });
    geojson = geojson.features.map(myturf);
    // 颜色比例尺
    const colorScale = d3.scaleSequential(
      [d3.min(Object.values(data)) * 0.8, d3.max(Object.values(data)) * 1.2],
      d3.interpolateOranges
    );

    // 散点大小比例尺
    const rScale = d3
      .scaleSequential()
      .domain(d3.extent(Object.values(data)))
      .range([4, 8]);

    d3.select(".map svg").remove();
    const map_svg = d3
      .select(".map")
      .append("svg")
      .attr("width", w)
      .attr("height", h)
      .attr("viewBox", [0, 0, w, h])
      .style("max-width", "100%");
    // .call(zoom);
    const path = d3.geoPath().projection(projection);

    const path_g = map_svg.append("g").attr("class", "map_path").call(zoom);
    path_g
      .selectAll("path")
      .data(geojson, (d) => d.properties.name)
      .join("path")
      .attr("stroke", "#798ee0")
      .attr("stroke-width", 0.5)
      .attr("fill", (d, i) => {
        let pathName = d.properties.name;
        let pathIdx = SENSORLIST_NAME_SIM.indexOf(pathName);
        if (pathIdx !== -1) {
          return COLORLIST[pathIdx];
        } else return "#eee";
      })
      .attr("id", (d) => d.properties.name)
      .attr("d", path)
      .on("mouseover", function (event, d) {
        // d3.select(this).style("stroke-width", 2);
      });

    const sensors_g = map_svg.append("g").attr("class", "map_sensor");
    const map_sensor_tooltip = d3
      .select("#map-sensor-tooltip")
      .style("opacity", 0);
    sensors_g
      .selectAll("circle")
      .data(sensors)
      .join("circle")
      .attr("class", "map-sensor-circle")
      .attr("cx", (d) => {
        return projection(d.slice(1))[0];
      })
      .attr("cy", (d) => projection(d.slice(1))[1])
      .attr("r", (d) => rScale(data[d[0]]))
      .attr("fill", (d, i) => colorScale(data[d[0]]))
      .on("mouseover", function (event, d) {
        map_sensor_tooltip.transition().duration(50).style("opacity", 1);
        d3.select("#map-sensor-tooltip")
          .text(d[0] + ": " + data[d[0]])
          .style("left", `${event.offsetX + 10}px`)
          .style("top", `${event.offsetY - 15}px`);
      })
      .on("mouseout", function (event, d) {
        map_sensor_tooltip.transition().duration(50).style("opacity", 0);
      });
  };

  const drawPunch = (punch, punch_max) => {
    if (punchRef.current === null) return;
    let existInstance = echarts.getInstanceByDom(punchRef.current);
    if (existInstance !== undefined) {
      echarts.dispose(existInstance);
    }
    const punchChart = echarts.init(punchRef.current);

    const option = {
      tooltip: {
        position: "top",
        formatter: function (params) {
          return (
            rangeBarList[params.value[0]] +
            " " +
            SENSORLIST_NAME_EN1[params.value[1]] +
            " " +
            params.value[2]
          );
        },
        confine: true,
        padding: 4,
        extraCssText: "width:100px",
      },
      grid: {
        top: "2%",
        left: 0,
        bottom: "1%",
        right: 10,
        containLabel: true,
      },
      xAxis: {
        type: "category",
        data: rangeBarList,
        boundaryGap: false,
        splitLine: {
          show: true,
        },
        axisLine: {
          show: false,
        },
      },
      yAxis: {
        show: true,
        type: "category",
        data: SENSORLIST_NAME_EN1,
        offset:5,
        axisLine: {
          show: false,
        },
        axisTick: {
          show: false,  
      },
      axisLabel: {
        show: true,  
    },
      },
      dataZoom: [
        {
          type: "inside",
          start: 0,
          end: 100,
        },
      ],
      series: [
        {
          name: "Punch Card",
          type: "scatter",
          symbolSize: function (val) {
            return (val[2] / punch_max[val[0]]) * 20;
          },
          data: punch,
          animationDelay: function (idx) {
            return idx * 5;
          },
          itemStyle: {
            color: (params) => {
              return COLORLIST[
                parseInt(params.dataIndex / rangeBarList.length)
              ];
            },
          },
        },
      ],
    };
    punchChart.setOption(option, true);

    // hover 时与 Map 组件高亮交互
    punchChart.on("mouseover", (e) => {
      // console.log("mouseover", e);

      senIdx.current = e.data[1];
      // senIdx.current = parseInt(e.dataIndex / rangeBarList.length);
      // let polIdx = e.dataIndex % rangeBarList.length;
      d3.selectAll(".map-sensor-circle")
        .filter((item) => {
          return item[0] === SENSORLIST_NAME[senIdx.current];
        })
        .attr("stroke", "#333")
        .attr("stroke-width", 3);
    });

    punchChart.on("mouseout", (e) => {
      d3.selectAll(".map-sensor-circle").attr("stroke-width", 0);
    });
  };

  return (
    <div className="map-wrap">
      <div className="map" ref={mapRef}>
        <div id="map-sensor-tooltip"></div>
      </div>
      <div className="map-punch" ref={punchRef}></div>
    </div>
  );
};

export default Map;
