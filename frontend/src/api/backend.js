import axios from "axios";

const API = axios.create({ baseURL: "http://127.0.0.1:8000" });

API.interceptors.request.use(config => {
  const token = localStorage.getItem("auth_token");
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

export const uploadData = (formData) =>
  API.post("/inference/run", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });

export const getPatients = () => API.get("/patients/all");
export const getResult = (id) => API.get(`/patients/${id}`);

// auth
export const authLogin = (payload) => API.post("/auth/login", payload);
export const authSignup = (payload) => API.post("/auth/signup", payload);
