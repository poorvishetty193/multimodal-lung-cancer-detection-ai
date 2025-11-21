import React, { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { useAuth } from "../auth/AuthContext";
import { motion } from "framer-motion";

export default function Login() {
  const { login } = useAuth();
  const nav = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [role, setRole] = useState("user");

  function submit(e) {
    e.preventDefault();
    login({ email, password, role });
    // redirect depending on role
    if (role === "admin") nav("/admin");
    else nav("/dashboard");
  }

  return (
    <div className="auth-page">
      <motion.div className="auth-card glass" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
        <div className="auth-left">
          <h2>Sign In</h2>
          <form onSubmit={submit} className="auth-form">
            <input value={email} onChange={(e) => setEmail(e.target.value)} placeholder="Email" required />
            <input value={password} type="password" onChange={(e) => setPassword(e.target.value)} placeholder="Password" required />
            <select value={role} onChange={(e) => setRole(e.target.value)}>
              <option value="user">Patient / User</option>
              <option value="doctor">Doctor</option>
              <option value="admin">Admin</option>
            </select>
            <button className="btn-primary" type="submit">Sign In</button>
          </form>
          <p className="muted">Don't have an account? <Link to="/signup">Sign up</Link></p>
        </div>

        <div className="auth-right">
          <div className="welcome">Hello, Friend!</div>
          <p className="muted">Register with your details to access all features.</p>
        </div>
      </motion.div>
    </div>
  );
}
