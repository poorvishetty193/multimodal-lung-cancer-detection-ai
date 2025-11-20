import { Link, useNavigate } from "react-router-dom";
import ThemeToggle from "./ThemeToggle";

export default function NavBar({ user, setUser }) {
  const navigate = useNavigate();

  function handleLogout() {
    // clear local user (and token) and redirect
    localStorage.removeItem("auth_token");
    setUser(null);
    navigate("/");
  }

  return (
    <nav className="w-full bg-white/80 dark:bg-slate-900/80 backdrop-blur-sm border-b dark:border-slate-700">
      <div className="max-w-6xl mx-auto px-4 py-3 flex items-center justify-between">
        <div className="flex items-center gap-4">
          <Link to="/" className="text-2xl font-extrabold tracking-tight">
            Lung<span className="text-blue-600">AI</span>
          </Link>
          <div className="hidden md:flex gap-3 text-sm text-slate-600 dark:text-slate-300">
            <Link to="/upload" className="hover:underline">Upload</Link>
            <Link to="/patients" className="hover:underline">Patients</Link>
            <Link to="/intro" className="hover:underline">About</Link>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <ThemeToggle />
          {user ? (
            <>
              <span className="text-sm text-slate-700 dark:text-slate-200 hidden sm:inline">
                {user.name || user.email}
              </span>
              <button
                onClick={handleLogout}
                className="bg-red-500 text-white px-3 py-1 rounded-md text-sm"
              >
                Logout
              </button>
            </>
          ) : (
            <>
              <Link to="/login" className="text-sm px-3 py-1 border rounded-md">Login</Link>
              <Link to="/signup" className="text-sm px-3 py-1 bg-blue-600 text-white rounded-md">Sign Up</Link>
            </>
          )}
        </div>
      </div>
    </nav>
  );
}
