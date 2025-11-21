import React from "react";
import { useLocation } from "react-router-dom";
import "../styles/auth.css";

export default function AuthLayout({ children }) {
  const location = useLocation();
  const isSignup = location.pathname === "/signup";

  return (
    <div className={`container ${isSignup ? "active" : ""}`}>
      <div className="curved-shape"></div>
      <div className="curved-shape2"></div>

      {/* LOGIN FORM (left) */}
      <div className="form-box Login">
        <h2 className="animation" style={{ ["--D"]: 0, ["--S"]: 21 }}>Login</h2>
        <form>{children?.loginForm}</form>
      </div>

      {/* Right info for Login */}
      <div className="info-content Login">
        <h2 className="animation" style={{ ["--D"]: 0, ["--S"]: 20 }}>WELCOME BACK!</h2>
        <p className="animation" style={{ ["--D"]: 1, ["--S"]: 21 }}>
          We are happy to have you with us again. If you need anything, we are here to help.
        </p>
      </div>

      {/* REGISTER FORM (right) */}
      <div className="form-box Register">
        <h2 className="animation" style={{ ["--li"]: 17, ["--S"]: 0 }}>Register</h2>
        <form>{children?.signupForm}</form>
      </div>

      {/* Left info for Register */}
      <div className="info-content Register">
        <h2 className="animation" style={{ ["--li"]: 17, ["--S"]: 0 }}>WELCOME!</h2>
        <p className="animation" style={{ ["--li"]: 18, ["--S"]: 1 }}>
          We're delighted to have you here. If you need any assistance, feel free to reach out.
        </p>
      </div>
    </div>
  );
}
