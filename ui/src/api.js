import axios from 'axios'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

// Submit scan (CT + audio + metadata)
export async function submitScan({ ctFile, audioFile, metadata }) {
  const form = new FormData()

  if (ctFile) form.append('file', ctFile)
  if (audioFile) form.append('audio_file', audioFile)

  form.append('metadata', JSON.stringify(metadata || {}))

  const url = `${API_BASE}/api/v1/scan/submit`
  const resp = await axios.post(url, form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 20000
  })

  return resp.data
}

// Fetch job status (status + progress + results if completed)
export async function getJob(jobId) {
  const url = `${API_BASE}/api/v1/job/${jobId}`
  const resp = await axios.get(url, { timeout: 10000 })
  return resp.data
}

// Fetch ONLY results (full output)
export async function getJobResults(jobId) {
  const url = `${API_BASE}/api/v1/job/${jobId}/results`
  const resp = await axios.get(url, { timeout: 10000 })
  return resp.data
}
