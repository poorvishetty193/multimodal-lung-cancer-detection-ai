import React from "react";
import { useAuth } from "../auth/AuthContext";

export default function AdminDashboard() {
  const { user } = useAuth();

  return (
    <div className="page container">
      <h2>Admin Panel</h2>
      <p className="muted">User management, logs, system status (stubs).</p>

      <div className="card-grid">
        <div className="card">
          <h3>Users</h3>
          <p>Manage users and roles (future CRUD).</p>
        </div>
        <div className="card">
          <h3>System</h3>
          <p>Jobs, inference queue and monitoring.</p>
        </div>
      </div>
    </div>
  );
}
