import React, { useState } from "react";
import AuthLayout from "./AuthLayout";
import { useNavigate, Link } from "react-router-dom";

export default function Login({ setUser }) {
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const loginForm = (
    <>
      <div className="input-box animation" style={{ ["--D"]: 1, ["--S"]: 22 }}>
        <input
          type="email"
          required
          value={email}
          onChange={(e) => setEmail(e.target.value)}
        />
        <label>Email</label>
      </div>

      <div className="input-box animation" style={{ ["--D"]: 2, ["--S"]: 23 }}>
        <input
          type="password"
          required
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />
        <label>Password</label>
      </div>

      <div className="input-box animation" style={{ ["--D"]: 3, ["--S"]: 24 }}>
        <button
          type="submit"
          className="btn"
          onClick={async (e) => {
            e.preventDefault();
            try {
              const res = await fetch("http://127.0.0.1:8000/auth/login", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ email, password }),
              });
              const data = await res.json();
              if (!res.ok) throw new Error(data.detail || "Login failed");
              localStorage.setItem("auth_token", data.token || "dummy-token");
              localStorage.setItem("auth_user", JSON.stringify(data.user || { email }));
              navigate("/");
            } catch (err) {
              alert(err.message || "Login error");
            }
          }}
        >
          Login
        </button>
      </div>

      <div className="regi-link animation" style={{ ["--D"]: 4, ["--S"]: 25 }}>
        <p>
          Don't have an account? <br />
          <Link to="/signup" className="SignUpLink">Sign Up</Link>

  return <AuthLayout children={{ loginForm }} />;
}
