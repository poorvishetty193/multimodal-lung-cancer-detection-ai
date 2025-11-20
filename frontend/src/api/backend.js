import axios from "axios";

const API = axios.create({ baseURL: "http://127.0.0.1:8000" });

export const uploadData = (formData) =>
  API.post("/inference/run", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });

export const getPatients = () => API.get("/patients/all");

export const getResult = (id) => API.get(`/patients/${id}`);