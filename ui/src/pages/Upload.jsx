import React, { useState } from "react";
import FileInput from "../components/FileInput";
import { submitScan } from "../api";

export default function Upload({ onJobCreated }) {
  const [ctFile, setCtFile] = useState(null);
  const [audioFile, setAudioFile] = useState(null);

  const [age, setAge] = useState("");
  const [packYears, setPackYears] = useState("");
  const [symptoms, setSymptoms] = useState("");

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  async function handleSubmit(e) {
    e.preventDefault();
    setError(null);

    if (!ctFile) {
      setError("Please upload a CT file or an image (PNG/JPG/ZIP).");
      return;
    }

    setLoading(true);

    try {
      const metadata = {
        age: age ? Number(age) : null,
        smoking_history_pack_years: packYears ? Number(packYears) : null,
        symptoms: symptoms || "",
      };

      // smoking input field must match backend param name
      const resp = await submitScan({
        ctFile,
        audioFile,
        metadata,
        smoking: packYears || "0", // IMPORTANT FIX
      });

      onJobCreated(resp.job_id);
    } catch (err) {
      console.error(err);
      setError(
        err?.response?.data?.detail ||
        "Upload failed. Please try again."
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
      <h2 className="text-xl font-semibold mb-4">Upload Scan</h2>

      <form onSubmit={handleSubmit} className="space-y-5">
        {/* CT or Image */}
        <FileInput
          label="CT / Image (PNG, JPG, JPEG, ZIP, DICOM-zip)"
          accept=".png,.jpg,.jpeg,.zip,application/zip,image/png,image/jpeg"
          onFile={setCtFile}
        />

        {/* Optional Audio */}
        <FileInput
          label="Audio (WAV / MP3 / M4A) — Optional"
          accept="audio/*"
          onFile={setAudioFile}
        />

        {/* Age + Smoking */}
        <div className="grid grid-cols-2 gap-3">
          
          <div>
            <label className="block text-sm mb-1">Age</label>
            <input
              type="number"
              value={age}
              onChange={(e) => setAge(e.target.value)}
              className="w-full rounded border px-3 py-2 bg-gray-50 dark:bg-gray-700"
              placeholder="e.g., 45"
            />
          </div>

          <div>
            <label className="block text-sm mb-1">Smoking Pack-Years</label>
            <input
              type="number"
              value={packYears}
              onChange={(e) => setPackYears(e.target.value)}
              className="w-full rounded border px-3 py-2 bg-gray-50 dark:bg-gray-700"
              placeholder="e.g., 10"
            />
          </div>

        </div>

        {/* Symptoms */}
        <div>
          <label className="block text-sm mb-1">Symptoms (optional)</label>
          <input
            value={symptoms}
            onChange={(e) => setSymptoms(e.target.value)}
            className="w-full rounded border px-3 py-2 bg-gray-50 dark:bg-gray-700"
            placeholder="e.g., chronic cough"
          />
        </div>

        {error && <div className="text-sm text-red-500">{error}</div>}

        <div className="flex items-center gap-3">
          <button
            disabled={loading}
            className="px-4 py-2 bg-indigo-600 text-white rounded disabled:opacity-50"
          >
            {loading ? "Uploading..." : "Submit Scan"}
          </button>

          <div className="text-sm text-gray-500">
            Your scan will be processed by the AI worker.
          </div>
        </div>
      </form>
    </section>
  );
}
