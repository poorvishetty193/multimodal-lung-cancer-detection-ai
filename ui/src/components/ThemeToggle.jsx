import React, { useEffect, useState } from 'react'
import clsx from 'clsx'

export default function ThemeToggle(){
  const [dark, setDark] = useState(() => {
    return document.documentElement.classList.contains('dark')
  })

  useEffect(() => {
    if (dark) document.documentElement.classList.add('dark')
    else document.documentElement.classList.remove('dark')
  }, [dark])

  return (
    <button
      aria-label="toggle theme"
      onClick={() => setDark(d => !d)}
      className={clsx('px-3 py-1 rounded-md border', dark ? 'bg-gray-800 border-gray-700 text-white' : 'bg-white border-gray-200')}
    >
      {dark ? 'Dark' : 'Light'}
    </button>
  )
}
