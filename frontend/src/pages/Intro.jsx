import React from "react";
import { Link } from "react-router-dom";

export default function Intro() {
  return (
    <div className="intro-hero">
      <div className="intro-card glass">
        <div className="intro-left">
          <h1>Lung Cancer Detection AI</h1>
          <p className="muted">
            Multimodal analysis: CT scans + lung audio + patient metadata. Fast, clinically-oriented preliminary reports to help
            clinicians triage and prioritize.
          </p>
          <div className="cta-row">
            <Link to="/signup" className="btn-primary">Create account</Link>
            <Link to="/login" className="btn-ghost">Login</Link>
          </div>
        </div>

        <div className="intro-right">
          {/* background accent uses uploaded screenshot */}
          <div className="screenshot" />
        </div>
      </div>
    </div>
  );
}
