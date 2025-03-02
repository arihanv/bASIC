"use client"

import { useEffect, useState } from "react"

const StatusBar = ({ gpuTemp, gpuLoad }: { gpuTemp: number; gpuLoad: number }) => {
  const [animatedTemp, setAnimatedTemp] = useState(0)
  const [animatedLoad, setAnimatedLoad] = useState(0)

  useEffect(() => {
    const tempInterval = setInterval(() => {
      setAnimatedTemp((prev) => {
        const diff = gpuTemp - prev
        return prev + diff * 0.1
      })
    }, 50)

    const loadInterval = setInterval(() => {
      setAnimatedLoad((prev) => {
        const diff = gpuLoad - prev
        return prev + diff * 0.1
      })
    }, 50)

    return () => {
      clearInterval(tempInterval)
      clearInterval(loadInterval)
    }
  }, [gpuTemp, gpuLoad])

  return (
    <div className="bg-black border-b border-green-900/30 p-2 text-xs flex justify-between items-center font-mono">
      <div className="flex items-center space-x-4">
        <div className="flex items-center">
          <span className="text-green-500">[TEMP]</span>
          <div className="ml-2 w-20 bg-green-900/30 h-4 rounded-sm overflow-hidden relative">
            <div
              className="h-full bg-gradient-to-r from-green-500 to-yellow-500 transition-all duration-300 ease-out absolute top-0 left-0"
              style={{ width: `${(animatedTemp / 100) * 100}%` }}
            ></div>
            <span className="absolute inset-0 flex items-center justify-center text-black font-bold mix-blend-difference">
              {Math.round(animatedTemp)}°C
            </span>
          </div>
        </div>
        <div className="flex items-center">
          <span className="text-green-500">[LOAD]</span>
          <div className="ml-2 w-20 bg-green-900/30 h-4 rounded-sm overflow-hidden relative">
            <div
              className="h-full bg-gradient-to-r from-green-500 to-red-500 transition-all duration-300 ease-out absolute top-0 left-0"
              style={{ width: `${animatedLoad}%` }}
            ></div>
            <span className="absolute inset-0 flex items-center justify-center text-black font-bold mix-blend-difference">
              {Math.round(animatedLoad)}%
            </span>
          </div>
        </div>
      </div>
      <div className="flex items-center space-x-2">
        <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></div>
        <span className="text-green-500">[QUANTUM CORES ACTIVE]</span>
      </div>
    </div>
  )
}

export default StatusBar

