import { Link } from "react-router-dom";

export default function Intro() {
  return (
    <div className="min-h-[70vh] flex items-center justify-center bg-gradient-to-b from-white to-slate-50 dark:from-slate-900 dark:to-slate-800">
      <div className="max-w-4xl px-6 py-16 text-center rounded-2xl shadow-lg bg-white/60 dark:bg-slate-900/60 backdrop-blur-md">
        <h1 className="text-4xl md:text-5xl font-extrabold mb-4">
          Multimodal Lung Cancer Detection AI
        </h1>
        <p className="text-slate-600 dark:text-slate-300 mb-6">
          Combine CT segmentation, audio analysis and clinical metadata into a single
          multimodal risk score — built for clinicians and research.
        </p>

        <div className="flex items-center justify-center gap-4">
          <Link to="/upload" className="px-6 py-3 bg-blue-600 text-white rounded-md shadow">
            Start Diagnosis
          </Link>
          <Link to="/patients" className="px-6 py-3 border rounded-md text-slate-700 dark:text-slate-200">
            Patient Records
          </Link>
        </div>

        <div className="mt-10 text-left text-sm text-slate-600 dark:text-slate-300 space-y-3">
          <p><b>How it works:</b> CT slices are preprocessed and segmented with a UNet, audio is analyzed with a CRNN and a fusion model combines embeddings and metadata to produce a risk score.</p>
          <p><b>Clinical use:</b> Intended for research / decision-support. Always confirm with a radiologist.</p>
        </div>
      </div>
    </div>
  );
}
