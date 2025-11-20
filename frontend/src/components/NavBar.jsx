import React from "react";
import { Link } from "react-router-dom";
import { motion } from "framer-motion";

export default function Navbar() {
  return (
    <motion.nav
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="fixed top-0 left-0 w-full backdrop-blur-xl bg-white/10 border-b border-white/20 shadow-lg z-50"
    >
      <div className="max-w-6xl mx-auto px-6 py-4 flex justify-between items-center">
        <Link to="/" className="text-2xl font-extrabold tracking-wide text-white">
          LungAI
        </Link>

        <div className="flex gap-8 text-gray-200 font-semibold">
          <Link to="/" className="hover:text-white transition">Home</Link>
          <Link to="/upload" className="hover:text-white transition">Upload</Link>
          <Link to="/patients" className="hover:text-white transition">Patients</Link>
          <Link to="/login" className="hover:text-white transition">Login</Link>
        </div>
      </div>
    </motion.nav>
  );
}
