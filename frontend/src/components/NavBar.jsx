import React from "react";
import { Link, useNavigate } from "react-router-dom";
import { useAuth } from "../auth/AuthContext";

export default function Navbar() {
  const { user, logout } = useAuth();
  const nav = useNavigate();

  function handleLogout() {
    logout();
    nav("/");
  }

  return (
    <header className="nav">
      <div className="nav-inner">
        <div className="brand">LungAI</div>
        <nav className="links">
          {user?.role === "admin" ? (
            <>
              <Link to="/admin">Admin</Link>
            </>
          ) : null}
          <Link to="/dashboard">Dashboard</Link>
          <Link to="/upload">Upload</Link>
          {user?.role === "doctor" || user?.role === "admin" ? <Link to="/patients">Patients</Link> : null}
          <button className="btn-ghost" onClick={handleLogout}>
            Logout
          </button>
        </nav>
      </div>
    </header>
  );
}
