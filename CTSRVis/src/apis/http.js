/**
 * 封装http请求
 */
import axios from "axios";
axios.defaults.baseURL = "/api";
axios.defaults.headers.post["Content-Type"] = "application/json; charset=UTF-8";
axios.defaults.timeout = 20000;
axios.interceptors.request.use((config) => {
  config.headers = { DeviceType: "H5" };
  return config;
});

export const get = (url, params) => {
  return new Promise((resolve, reject) => {
    axios
      .get(url, {
        params: params,
      })
      .then((res) => {
        resolve(res.data);
      })
      .catch((err) => {
        reject(err.data);
      });
  });
};

export const post = (url, params) => {
  return new Promise((resolve, reject) => {
    axios
      .post(url, params)
      .then((res) => {
        resolve(res.data);
      })
      .catch((err) => {
        reject(err.data);
      });
  });
};
