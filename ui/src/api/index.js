import axios from "axios";

const API = import.meta.env.VITE_API_BASE || "http://localhost:8000";

// ---------------------------------------------------------
//  SUBMIT SCAN (CT or PNG/JPG + optional audio + metadata)
// ---------------------------------------------------------
export async function submitScan({ ctFile, audioFile, metadata }) {
  const form = new FormData();

  // main file (CT or Image)
  form.append("file", ctFile);

  // optional audio
  if (audioFile) form.append("audio_file", audioFile);

  // metadata JSON string
  form.append("metadata", JSON.stringify(metadata));

  // backend expects a **separate** form field for pack-years
  form.append(
    "smoking_pack_years",
    metadata.smoking_history_pack_years ?? "0"
  );

  const res = await axios.post(`${API}/api/v1/scan/submit`, form, {
    headers: { "Content-Type": "multipart/form-data" },
  });

  return res.data;
}

// ---------------------------------------------------------
//  GET JOB STATUS
// ---------------------------------------------------------
export async function getJob(id) {
  const res = await axios.get(`${API}/api/v1/job/${id}`);
  return res.data;
}

// ---------------------------------------------------------
//  GET FINAL RESULTS
// ---------------------------------------------------------
export async function getJobResults(id) {
  const res = await axios.get(`${API}/api/v1/job/${id}/results`);
  return res.data;
}
