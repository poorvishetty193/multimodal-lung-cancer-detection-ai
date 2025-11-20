import { useContext } from "react";
import { ThemeContext } from "../context/ThemeContext";

export default function ThemeToggle() {
  const { theme, toggleTheme } = useContext(ThemeContext);

  return (
    <button
      onClick={toggleTheme}
      className="fixed bottom-6 right-6 p-3 rounded-full shadow-lg
                 bg-gray-200 dark:bg-gray-700 dark:text-white transition"
    >
      {theme === "light" ? "🌙" : "☀️"}
    </button>
  );
}
