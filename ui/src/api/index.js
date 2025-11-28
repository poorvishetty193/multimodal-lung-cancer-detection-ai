import axios from "axios";

const API = import.meta.env.VITE_API_BASE || "http://localhost:8000";

export async function submitScan({ ctFile, audioFile, metadata, smoking }) {
  const form = new FormData();

  form.append("file", ctFile);
  if (audioFile) form.append("audio_file", audioFile);

  form.append("metadata", JSON.stringify(metadata));
  form.append("smoking_pack_years", smoking || "0"); // IMPORTANT FIX

  const res = await axios.post(`${API}/api/v1/scan/submit`, form, {
    headers: { "Content-Type": "multipart/form-data" },
  });

  return res.data;
}

export async function getJob(id) {
  const res = await axios.get(`${API}/api/v1/job/${id}`);
  return res.data;
}

export async function getJobResults(id) {
  const res = await axios.get(`${API}/api/v1/job/${id}/results`);
  return res.data;
}
