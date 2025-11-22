import React from 'react'

function ProbRow({ label, value }){
  return (
    <div className="flex justify-between py-1 border-b border-dashed border-gray-200 dark:border-gray-700">
      <div className="text-sm">{label}</div>
      <div className="font-semibold">{(value*100).toFixed(1)}%</div>
    </div>
  )
}

export default function Results({ results, onBack }){
  // results has ct, audio, metadata, fusion
  const ct = results.ct || {}
  const audio = results.audio || {}
  const meta = results.metadata || {}
  const fusion = results.fusion || results

  const final = fusion.final_probs || {}

  return (
    <section className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
      <div className="flex items-start justify-between">
        <div>
          <h2 className="text-xl font-semibold mb-1">Results</h2>
          <div className="text-sm text-gray-500">Final fusion risk: <strong>{fusion.risk_score?.toFixed(3) ?? 'N/A'}</strong></div>
        </div>
        <div>
          <button onClick={onBack} className="px-3 py-1 bg-gray-200 dark:bg-gray-700 rounded">Back</button>
        </div>
      </div>

      <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="col-span-1 bg-gray-50 dark:bg-gray-900 p-3 rounded">
          <h3 className="font-semibold mb-2">Fusion (final)</h3>
          {Object.keys(final).length ? (
            <div>
              {Object.entries(final).map(([k,v]) => <ProbRow key={k} label={k} value={v} />)}
            </div>
          ) : <div className="text-sm text-gray-500">No final probs</div>}
        </div>

        <div className="col-span-1 bg-gray-50 dark:bg-gray-900 p-3 rounded">
          <h3 className="font-semibold mb-2">CT output</h3>
          <div className="text-sm mb-2">Nodules:</div>
          {ct.nodules && ct.nodules.length ? (
            ct.nodules.map((n, idx) => (
              <div key={idx} className="p-2 mb-2 rounded border border-gray-200 dark:border-gray-700">
                <div className="text-sm font-medium">Nodule #{idx+1} — {Math.round(n.diameter_mm)} mm</div>
                <div className="text-xs text-gray-500">confidence: {n.confidence}</div>
                <div className="mt-2">
                  {Object.entries(n.nodule_probs).map(([k,v]) => <div key={k} className="text-xs">{k}: {(v*100).toFixed(1)}%</div>)}
                </div>
              </div>
            ))
          ) : <div className="text-sm text-gray-500">No nodules</div>}
        </div>

        <div className="col-span-1 bg-gray-50 dark:bg-gray-900 p-3 rounded">
          <h3 className="font-semibold mb-2">Audio & Metadata</h3>
          <div className="mb-2">
            <div className="text-sm mb-1 font-medium">Audio probs</div>
            {audio.audio_probs ? Object.entries(audio.audio_probs).map(([k,v]) => <div key={k} className="text-xs">{k}: {(v*100).toFixed(1)}%</div>) : <div className="text-sm text-gray-500">No audio</div>}
          </div>
          <div>
            <div className="text-sm mb-1 font-medium">Metadata probs</div>
            {meta.metadata_probs ? Object.entries(meta.metadata_probs).map(([k,v]) => <div key={k} className="text-xs">{k}: {(v*100).toFixed(1)}%</div>) : <div className="text-sm text-gray-500">No metadata</div>}
          </div>
        </div>
      </div>

      <div className="mt-6 text-xs text-gray-500">
        Embeddings and full JSON available in API results (this UI shows a digest). Extend to visualize embeddings with a chart library.
      </div>
    </section>
  )
}
