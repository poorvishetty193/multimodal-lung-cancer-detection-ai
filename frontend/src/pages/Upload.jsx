import React, { useState } from "react";

export default function Upload() {
  const [ctZip, setCtZip] = useState(null);
  const [audio, setAudio] = useState(null);
  const [age, setAge] = useState("");
  const [sex, setSex] = useState("");

  function handleSubmit(e) {
    e.preventDefault();
    // TODO: send multipart to backend (axios)
    alert("Uploaded (stub). Implement backend integration.");
  }

  return (
    <div className="page container">
      <h2>Upload CT & Audio</h2>
      <form className="form card" onSubmit={handleSubmit}>
        <label>CT folder (zip)</label>
        <input type="file" accept=".zip" onChange={(e) => setCtZip(e.target.files[0])} />

        <label>Audio (.wav)</label>
        <input type="file" accept=".wav" onChange={(e) => setAudio(e.target.files[0])} />

        <label>Age</label>
        <input type="number" value={age} onChange={(e) => setAge(e.target.value)} />

        <label>Sex</label>
        <select value={sex} onChange={(e) => setSex(e.target.value)}>
          <option value="">Select</option>
          <option value="female">Female</option>
          <option value="male">Male</option>
        </select>

        <div className="row">
          <button className="btn-primary" type="submit">Run Analysis</button>
        </div>
      </form>
    </div>
  );
}
