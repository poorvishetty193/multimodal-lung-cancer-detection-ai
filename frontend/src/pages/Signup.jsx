import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { authSignup } from "../api/backend";

export default function Signup() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const navigate = useNavigate();

  async function submit(e) {
    e.preventDefault();
    try {
      await authSignup({ name, email, password });
      alert("Account created. Please login.");
      navigate("/login");
    } catch (err) {
      alert(err?.response?.data?.message || "Signup failed");
    }
  }

  return (
    <div className="max-w-md mx-auto mt-16 p-6 bg-white/60 dark:bg-slate-900/60 rounded-xl shadow-lg backdrop-blur">
      <h2 className="text-2xl font-bold mb-4">Create account</h2>
      <form onSubmit={submit} className="space-y-4">
        <input className="w-full p-3 border rounded" placeholder="Full name" value={name} onChange={(e)=>setName(e.target.value)} />
        <input className="w-full p-3 border rounded" placeholder="Email" value={email} onChange={(e)=>setEmail(e.target.value)} />
        <input type="password" className="w-full p-3 border rounded" placeholder="Password" value={password} onChange={(e)=>setPassword(e.target.value)} />
        <button className="w-full bg-green-600 text-white p-3 rounded">Create account</button>
      </form>
    </div>
  );
}
