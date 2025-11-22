import React, { useState } from 'react'
import FileInput from '../components/FileInput'
import { submitScan } from '../api'

export default function Upload({ onJobCreated }){
  const [ctFile, setCtFile] = useState(null)
  const [audioFile, setAudioFile] = useState(null)
  const [age, setAge] = useState(50)
  const [packYears, setPackYears] = useState(0)
  const [symptoms, setSymptoms] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  async function handleSubmit(e){
    e.preventDefault()
    setError(null)
    setLoading(true)
    try {
      const metadata = { age: Number(age), smoking_history_pack_years: Number(packYears), symptoms }
      const resp = await submitScan({ ctFile, audioFile, metadata })
      onJobCreated(resp.job_id)
    } catch (err){
      console.error(err)
      setError(err?.response?.data?.detail || err.message || 'Upload failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <section className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
      <h2 className="text-xl font-semibold mb-4">Upload scan</h2>

      <form onSubmit={handleSubmit} className="space-y-4">
        <FileInput label="CT archive (zip/dicom-zip)" accept=".zip,application/zip" onFile={setCtFile} />
        <FileInput label="Audio (wav/mp3/m4a) — optional" accept="audio/*" onFile={setAudioFile} />

        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="block text-sm mb-1">Age</label>
            <input type="number" value={age} onChange={e=>setAge(e.target.value)} className="w-full rounded border px-3 py-2 bg-gray-50 dark:bg-gray-700" />
          </div>
          <div>
            <label className="block text-sm mb-1">Smoking pack-years</label>
            <input type="number" value={packYears} onChange={e=>setPackYears(e.target.value)} className="w-full rounded border px-3 py-2 bg-gray-50 dark:bg-gray-700" />
          </div>
        </div>

        <div>
          <label className="block text-sm mb-1">Symptoms (free text)</label>
          <input value={symptoms} onChange={e=>setSymptoms(e.target.value)} className="w-full rounded border px-3 py-2 bg-gray-50 dark:bg-gray-700" />
        </div>

        {error && <div className="text-sm text-red-500">{JSON.stringify(error)}</div>}

        <div className="flex items-center gap-3">
          <button disabled={loading} className="px-4 py-2 bg-indigo-600 text-white rounded">
            {loading ? 'Uploading...' : 'Submit Scan'}
          </button>
          <div className="text-sm text-gray-500">After submit, the worker will process the job. You will get a Job ID to poll.</div>
        </div>
      </form>
    </section>
  )
}
