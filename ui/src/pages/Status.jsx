import React, { useEffect, useState } from "react";
import { getJob, getJobResults } from "../api";

export default function Status({ jobId, onComplete }) {
  const [status, setStatus] = useState(null);
  const [progress, setProgress] = useState(0);
  const [polling, setPolling] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let mounted = true;

    async function check() {
      try {
        const data = await getJob(jobId);
        if (!mounted) return;

        setStatus(data.status);
        setProgress(Number(data.progress || 0));

        if (data.status === "completed") {
          // Use results directly from getJob API response
          const cleanResults = data.results ? data.results : {};

          setPolling(false);
          onComplete(cleanResults);
        }

        if (data.status === "failed") {
          setPolling(false);
          setError("Processing failed");
        }
      } catch (err) {
        console.error(err);
        // Handle 404 as no results yet, call onComplete with empty results and clear error
        if (err.response && err.response.status === 404) {
          setError(null);
          setStatus("not found");
          setPolling(false);
          onComplete({});
        } else if (err.response && err.response.status) {
          setError(`Error ${err.response.status}: ${err.response.statusText}`);
        } else if (err.message) {
          setError(`Error: ${err.message}`);
        } else {
          setError("Could not fetch job status");
        }
      }
    }

    check();

    if (polling) {
      const id = setInterval(check, 2000);
      return () => {
        mounted = false;
        clearInterval(id);
      };
    }

    return () => {
      mounted = false;
    };
  }, [jobId, polling, onComplete]);

  return (
    <section className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
      <h2 className="text-xl font-semibold mb-4">Job status</h2>

      <div className="mb-3">
        <div className="text-sm text-gray-600 dark:text-gray-300">Job ID</div>
        <div className="font-mono text-sm">{jobId}</div>
      </div>

      <div className="mb-4">
        <div className="text-sm">
          Status: <strong>{status || "loading"}</strong>
        </div>

        <div className="w-full bg-gray-200 dark:bg-gray-700 rounded h-3 mt-2 overflow-hidden">
          <div
            style={{ width: `${progress}%` }}
            className="h-3 bg-indigo-600 transition-all"
          ></div>
        </div>

        <div className="text-sm text-gray-500 mt-1">{progress}%</div>
      </div>

      {error && <div className="text-red-500">{error}</div>}

      <button
        onClick={() => setPolling((v) => !v)}
        className="px-3 py-1 rounded border"
      >
        {polling ? "Pause Polling" : "Resume Polling"}
      </button>
    </section>
  );
}
