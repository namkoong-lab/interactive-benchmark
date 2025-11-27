import { Routes, Route } from "react-router-dom";
import Hero from "./components/Hero";
import Home from "./pages/Home";
import Submit from "./pages/Submit";
import Leaderboard from "./pages/Leaderboard";

function App() {
  return (
    <div className="min-h-screen">
      {/* Hero displays on all pages */}
      <Hero />

      <main>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/submit" element={<Submit />} />
          <Route path="/leaderboard" element={<Leaderboard />} />
        </Routes>
      </main>
    </div>
  );
}

export default App;
