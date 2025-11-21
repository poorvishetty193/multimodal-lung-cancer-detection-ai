import React from "react";
import { useAuth } from "../auth/AuthContext";

export default function UserDashboard() {
  const { user } = useAuth();

  return (
    <div className="page container">
      <h2>Welcome back, {user?.name}</h2>
      <p className="muted">Run a new analysis or view your past results.</p>

      <div className="card-grid">
        <div className="card">
          <h3>Run New Analysis</h3>
          <p>Upload CT & audio to perform multimodal inference.</p>
          <a className="btn-primary" href="/upload">Start</a>
        </div>

        <div className="card">
          <h3>My Reports</h3>
          <p>View previously generated reports.</p>
          <a className="btn-ghost" href="/patients">View</a>
        </div>
      </div>
    </div>
  );
}
