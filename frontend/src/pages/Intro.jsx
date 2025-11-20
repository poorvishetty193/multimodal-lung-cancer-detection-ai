import React from "react";
import { motion } from "framer-motion";
import { Link } from "react-router-dom";

export default function Introduction() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-black to-slate-800 text-white p-10 flex items-center justify-center">
      <motion.div
        initial={{ opacity: 0, y: 40 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.7 }}
        className="max-w-3xl text-center p-10 rounded-3xl bg-white/10 backdrop-blur-xl border border-white/20 shadow-2xl"
      >
        <h1 className="text-5xl font-extrabold mb-6">Lung Cancer Detection AI</h1>

        <p className="text-gray-300 text-lg leading-relaxed mb-8">
          This cutting-edge AI platform analyzes **CT scans**, **lung audio**, and **patient metadata** to provide an accurate and early
          prediction of potential lung cancer risk. Our system uses multimodal deep learning pipelines—UNet segmentation, 3D CT
          classification, CRNN audio modeling, and fusion neural networks—to deliver a detailed medical-style report.
        </p>

        <p className="text-gray-300 text-lg leading-relaxed mb-8">
          Built for doctors, radiologists, and research professionals, this tool enhances diagnostic speed, improves reliability,
          and ensures consistent assessment to support early detection.
        </p>

        <motion.div whileHover={{ scale: 1.05 }}>
          <Link
            to="/login"
            className="px-8 py-4 bg-blue-600 hover:bg-blue-700 rounded-xl text-lg font-semibold shadow-lg"
          >
            Get Started
          </Link>
        </motion.div>
      </motion.div>
    </div>
  );
}