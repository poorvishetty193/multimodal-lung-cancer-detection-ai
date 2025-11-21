import React from "react";
import { Routes, Route, Navigate } from "react-router-dom";

import Intro from "./pages/Intro";
import Login from "./pages/Login";
import Signup from "./pages/Signup";
import UserDashboard from "./pages/UserDashboard";
import AdminDashboard from "./pages/AdminDashboard";
import Upload from "./pages/Upload";
import Patients from "./pages/Patients";
import Results from "./pages/Results";

import Navbar from "./components/Navbar";
import { useAuth } from "./auth/AuthContext";

/**
 * RequireAuth component - wraps protected routes
 */
function RequireAuth({ children, allowedRoles }) {
  const { user } = useAuth();
  if (!user) return <Navigate to="/login" replace />;
  if (allowedRoles && !allowedRoles.includes(user.role)) return <Navigate to="/" replace />;
  return children;
}

export default function App() {
  const { user } = useAuth();

  return (
    <div className={user ? "app logged-in" : "app logged-out"}>
      {/* Navbar only when logged in */}
      {user && <Navbar />}

      <Routes>
        {/* public */}
        <Route path="/" element={<Intro />} />
        <Route path="/login" element={<Login />} />
        <Route path="/signup" element={<Signup />} />

        {/* protected - user */}
        <Route
          path="/dashboard"
          element={
            <RequireAuth allowedRoles={["user", "doctor"]}>
              <UserDashboard />
            </RequireAuth>
          }
        />

        {/* protected - admin */}
        <Route
          path="/admin"
          element={
            <RequireAuth allowedRoles={["admin"]}>
              <AdminDashboard />
            </RequireAuth>
          }
        />

        {/* shared protected */}
        <Route
          path="/upload"
          element={
            <RequireAuth allowedRoles={["user", "doctor", "admin"]}>
              <Upload />
            </RequireAuth>
          }
        />
        <Route
          path="/patients"
          element={
            <RequireAuth allowedRoles={["doctor", "admin"]}>
              <Patients />
            </RequireAuth>
          }
        />
        <Route
          path="/results/:id"
          element={
            <RequireAuth allowedRoles={["user", "doctor", "admin"]}>
              <Results />
            </RequireAuth>
          }
        />

        {/* fallback */}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </div>
  );
}
