import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../auth/AuthContext";
import { motion } from "framer-motion";

export default function Signup() {
  const { signup } = useAuth();
  const nav = useNavigate();
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [role, setRole] = useState("user");

  function submit(e) {
    e.preventDefault();
    signup({ name, email, password, role });
    if (role === "admin") nav("/admin");
    else nav("/dashboard");
  }

  return (
    <div className="auth-page">
      <motion.div className="auth-card glass" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
        <div className="auth-left">
          <h2>Create Account</h2>
          <form onSubmit={submit} className="auth-form">
            <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Full name" required />
            <input value={email} onChange={(e) => setEmail(e.target.value)} placeholder="Email" required />
            <input value={password} onChange={(e) => setPassword(e.target.value)} type="password" placeholder="Password" required />
            <select value={role} onChange={(e) => setRole(e.target.value)}>
              <option value="user">Patient / User</option>
              <option value="doctor">Doctor</option>
              <option value="admin">Admin</option>
            </select>
            <button className="btn-primary" type="submit">Sign Up</button>
          </form>
        </div>
        <div className="auth-right">
          <div className="welcome">Welcome!</div>
          <p className="muted">Use a valid email for verification (if enabled later).</p>
        </div>
      </motion.div>
    </div>
  );
}
