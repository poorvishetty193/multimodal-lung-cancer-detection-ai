import React from 'react'

export default function FileInput({ label, accept, onFile }){
  return (
    <label className="block">
      <div className="text-sm font-medium mb-1">{label}</div>
      <input
        type="file"
        accept={accept}
        onChange={(e) => onFile(e.target.files && e.target.files[0])}
        className="block w-full text-sm text-gray-700 dark:text-gray-200"
      />
    </label>
  )
}
