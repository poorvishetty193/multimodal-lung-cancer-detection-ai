import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { authLogin } from "../api/backend";

export default function Login({ setUser }) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  async function submit(e) {
    e.preventDefault();
    setLoading(true);
    try {
      const res = await authLogin({ email, password });
      // expect { token, user } from backend
      const { token, user } = res.data;
      localStorage.setItem("auth_token", token);
      setUser(user);
      navigate("/patients");
    } catch (err) {
      alert(err?.response?.data?.message || "Login failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="max-w-md mx-auto mt-16 p-6 bg-white/60 dark:bg-slate-900/60 rounded-xl shadow-lg backdrop-blur">
      <h2 className="text-2xl font-bold mb-4">Login</h2>
      <form onSubmit={submit} className="space-y-4">
        <input className="w-full p-3 border rounded" placeholder="Email" value={email} onChange={(e)=>setEmail(e.target.value)} />
        <input type="password" className="w-full p-3 border rounded" placeholder="Password" value={password} onChange={(e)=>setPassword(e.target.value)} />
        <button disabled={loading} className="w-full bg-blue-600 text-white p-3 rounded">
          {loading ? "Signing in..." : "Sign in"}
        </button>
      </form>

      <p className="mt-4 text-sm text-slate-600 dark:text-slate-300">
        New? <a href="/signup" className="text-blue-600">Create an account</a>
      </p>
    </div>
  );
}
