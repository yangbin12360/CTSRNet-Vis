import { Routes, Route } from "react-router-dom";

import "./App.less";
import HomePage from "./pages/homepage";
import AnalysisPage from "./pages/analysispage";
import DenseLine from "./components/DenseLine";
import Map from "./components/Map";
import ClusterPage from "./components/ClusterPage";
import RangeBar from "./components/RangeBar";
import ContrastHeat from "./components/ContrastHeat";

function App() {
  return (
    <div className="App">
      <Routes>
        <Route key="index" path="/" element={<HomePage />} />
        <Route key="analysis" path="/analysis" element={<AnalysisPage />} />
        <Route key="denseline" path="/denseline" element={<DenseLine />} />
        <Route key="map" path="/map" element={<Map />} />
        <Route key="cluster" path="/cluster" element={<ClusterPage />} />
        <Route key="rangebar" path="/rangebar" element={<RangeBar />} />
        <Route key="contrastheat" path="/contrastheat" element={<ContrastHeat />} />
      </Routes>
    </div>
  );
}

export default App;
