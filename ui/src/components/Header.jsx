import React from 'react'
import ThemeToggle from './ThemeToggle'

export default function Header(){
  return (
    <header className="border-b border-gray-200 dark:border-gray-800">
      <div className="max-w-5xl mx-auto px-6 py-4 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold">Lung Cancer Detection AI</h1>
          <p className="text-sm text-gray-600 dark:text-gray-400">Upload CT (zip) + audio to run the multimodal pipeline</p>
        </div>
        <div className="flex items-center gap-4">
          <a className="text-sm px-3 py-1 rounded-md bg-indigo-600 text-white" href="http://localhost:8000/docs" target="_blank" rel="noreferrer">API Docs</a>
          <ThemeToggle />
        </div>
      </div>
    </header>
  )
}
