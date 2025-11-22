import React, { useState } from 'react'
import Upload from './pages/Upload'
import Status from './pages/Status'
import Results from './pages/Results'
import Header from './components/Header'

export default function App(){
  const [jobId, setJobId] = useState(null)
  const [results, setResults] = useState(null)

  return (
    <div className="min-h-screen transition-colors duration-200 bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-gray-100">
      <Header />
      <main className="max-w-5xl mx-auto p-6">
        {!jobId && !results && (
          <Upload onJobCreated={(id)=> setJobId(id)} />
        )}

        {jobId && !results && (
          <Status jobId={jobId} onComplete={(res)=> { setResults(res); setJobId(null) }} />
        )}

        {results && (
          <Results results={results} onBack={()=> { setResults(null) }} />
        )}
      </main>
    </div>
  )
}
