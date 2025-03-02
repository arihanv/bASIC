"use client"

import { useEffect, useState } from "react"

const GPUModal = ({ onClose }: { onClose: () => void }) => {
  const [visible, setVisible] = useState(true)

  useEffect(() => {
    setVisible(true)
  }, [])

  if (!visible) return null

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-black border border-green-500 p-8 rounded-lg max-w-2xl w-full">
        <pre className="text-green-500 text-xs leading-tight">
          {`
   ┌───────────────────────────────────┐
   │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐  │
   │  │     │ │     │ │     │ │     │  │
   │  │ GPU │ │ GPU │ │ GPU │ │ GPU │  │
   │  │     │ │     │ │     │ │     │  │
   │  └─────┘ └─────┘ └─────┘ └─────┘  │
   │                                   │
   │    ┌───────────────────────┐      │
   │    │      VRAM 32 GB       │      │
   │    └───────────────────────┘      │
   │                                   │
   │  ┌───────────────────────────┐    │
   │  │     CUDA Cores 10496      │    │
   │  └───────────────────────────┘    │
   │                                   │
   └───────────────────────────────────┘
`}
        </pre>
        <button
          onClick={onClose}
          className="mt-4 bg-green-700 hover:bg-green-600 text-black font-bold py-2 px-4 rounded transition duration-300 ease-in-out transform hover:scale-105"
        >
          Get Started
        </button>
      </div>
    </div>
  )
}

export default GPUModal

