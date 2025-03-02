"use client"

import { useRef, useEffect, useState } from "react"
import { cn } from "@/lib/utils"
import { Zap } from "lucide-react"

interface ChatWindowProps {
  title: string
  description: string
  messages: { role: string; content: string; isStreaming?: boolean }[]
  className?: string
  isSpeculative?: boolean
  isStreaming: boolean
}

export default function ChatWindow({
  title,
  description,
  messages,
  className,
  isSpeculative = false,
  isStreaming,
}: ChatWindowProps) {
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const messagesContainerRef = useRef<HTMLDivElement>(null)
  const [prevMessagesLength, setPrevMessagesLength] = useState(messages.length)
  const [shouldAutoScroll, setShouldAutoScroll] = useState(true)
  const [lastContentLength, setLastContentLength] = useState("")
  
  // Detect user scrolling to prevent auto-scroll when user is viewing history
  useEffect(() => {
    const container = messagesContainerRef.current
    if (!container) return
    
    const handleScroll = () => {
      // Check if user has scrolled up
      const isScrolledToBottom = 
        container.scrollHeight - container.clientHeight <= container.scrollTop + 50
      setShouldAutoScroll(isScrolledToBottom)
    }
    
    container.addEventListener('scroll', handleScroll)
    return () => container.removeEventListener('scroll', handleScroll)
  }, [])
  
  // Get content to track changes
  const messagesContent = messages.map(m => m.content).join('')

  // Scroll to bottom when messages change or content is updated, but only if shouldAutoScroll is true
  useEffect(() => {
    // If there are new messages, always scroll
    const hasNewMessages = messages.length > prevMessagesLength
    // If content length changed significantly, it's likely a meaningful update
    const hasSignificantContentUpdate = messagesContent.length > lastContentLength.length + 100
    
    const shouldScroll = shouldAutoScroll || hasNewMessages || hasSignificantContentUpdate
    
    if (shouldScroll) {
      const scrollToBottom = () => {
        if (messagesEndRef.current) {
          messagesEndRef.current.scrollIntoView({ behavior: "smooth" })
          // Re-enable auto-scroll if we've explicitly scrolled to bottom
          setShouldAutoScroll(true)
        }
      }
      
      scrollToBottom()
      
      // Also set a small timeout to ensure scroll happens after content renders
      const timeoutId = setTimeout(scrollToBottom, 100)
      return () => clearTimeout(timeoutId)
    }
  }, [messages.length, messagesContent, shouldAutoScroll, prevMessagesLength, lastContentLength])

  // Update tracking variables
  useEffect(() => {
    setPrevMessagesLength(messages.length)
    setLastContentLength(messagesContent)
  }, [messages.length, messagesContent])

  return (
    <div className={cn("flex flex-col h-[calc(100vh-12rem)]", className)}>
      <div className="p-3 border-b border-green-900/30 flex items-center justify-between">
        <div>
          <h2 className="font-bold text-sm tracking-widest">{title}</h2>
          <p className="text-xs text-green-700 font-mono">{description}</p>
        </div>
        {isSpeculative && (
          <div className="flex items-center gap-1 bg-black border border-green-900/50 px-2 py-0.5 text-xs text-green-400">
            <Zap className="h-3 w-3" />
            <span>ACCELERATED</span>
          </div>
        )}
      </div>

      <div 
        ref={messagesContainerRef}
        className="flex-1 overflow-y-auto p-4 space-y-4 font-mono scrollbar-thin scrollbar-thumb-green-900 scrollbar-track-black"
        style={{ height: 'calc(100% - 8rem)', maxHeight: 'calc(100vh - 16rem)' }}
      >
        {messages.map((message, index) => {
          const isUser = message.role === "user"
          const isAssistant = message.role === "assistant" || message.role === "assistant-fast"

          return (
            <div key={`msg-${index}-${message.role}`} className={cn("flex flex-col max-w-[95%] space-y-1", isUser ? "ml-auto" : "mr-auto")}>
              <div className="text-xs text-green-700">{isUser ? "USER@forge:~$" : "MODEL@forge:~$"}</div>
              <div
                className={cn(
                  "px-3 py-2 text-sm leading-relaxed",
                  isUser
                    ? "bg-green-900/20 border border-green-900/30 text-green-400"
                    : isSpeculative
                      ? "text-green-500 border-l-2 border-green-700/50 pl-3"
                      : "text-green-500 border-l-2 border-green-700/30 pl-3",
                )}
              >
                {message.isStreaming ? (
                  <>
                    {message.content}
                    <span className="inline-block w-2 h-4 bg-green-500 ml-1 animate-pulse" />
                  </>
                ) : (
                  message.content || (
                    <div className="flex flex-col gap-2">
                      <div className="flex items-center gap-2">
                        <div className="h-4 w-4 rounded-full bg-green-500 animate-pulse" />
                        <span>Processing query...</span>
                      </div>
                      <div className="w-full bg-green-900/30 h-2 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-green-500 transition-all duration-500 ease-in-out animate-pulse"
                          style={{ width: "0%" }}
                        />
                      </div>
                    </div>
                  )
                )}
              </div>
            </div>
          )
        })}
        <div ref={messagesEndRef} />
      </div>
      {isStreaming && (
        <div className="p-2 border-t border-green-900/30 text-xs text-green-500 animate-pulse">
          GPU Cores Active | Processing...
        </div>
      )}
    </div>
  )
}

