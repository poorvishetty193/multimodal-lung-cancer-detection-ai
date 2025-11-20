import { BrowserRouter, Routes, Route } from "react-router-dom";
import Home from "./pages/Home";
import Upload from "./pages/Upload";
import Patients from "./pages/Patients";
import Results from "./pages/Results";
import Intro from "./pages/Intro";
import Login from "./pages/Login";
import Signup from "./pages/Signup";
import NavBar from "./components/NavBar";
import { useEffect, useState } from "react";

export default function App() {
  const [user, setUser] = useState(null);

  useEffect(() => {
    // load user if token exists (you might call /auth/me)
    const token = localStorage.getItem("auth_token");
    if (token) {
      // for now simple placeholder: we store minimal user in localStorage if you want
      const u = JSON.parse(localStorage.getItem("auth_user") || "null");
      setUser(u);
    }
  }, []);

  return (
    <BrowserRouter>
      <NavBar user={user} setUser={setUser} />
      <div className="mt-6">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/intro" element={<Intro />} />
          <Route path="/upload" element={<Upload />} />
          <Route path="/patients" element={<Patients />} />
          <Route path="/results/:id" element={<Results />} />
          <Route path="/login" element={<Login setUser={setUser} />} />
          <Route path="/signup" element={<Signup />} />
        </Routes>
      </div>
    </BrowserRouter>
  );
}
