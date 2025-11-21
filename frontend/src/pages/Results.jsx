import React from "react";
import { useParams } from "react-router-dom";

export default function Results() {
  const { id } = useParams();
  return (
    <div className="page container">
      <h2>Report: {id}</h2>
      <pre className="report card">Report content (stub). Replace with backend fetch by id.</pre>
    </div>
  );
}
