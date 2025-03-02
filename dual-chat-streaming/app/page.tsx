"use client";

import type React from "react";

import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Send } from "lucide-react";
import ChatWindow from "@/components/chat-window";
import GPUModal from "@/components/gpu-modal";
import StatusBar from "@/components/status-bar";

export default function Home() {
	const [messages, setMessages] = useState<
		{ role: string; content: string; isStreaming?: boolean }[]
	>([]);
	const [isStreaming, setIsStreaming] = useState(false);
	const [input, setInput] = useState("");
	const [showModal, setShowModal] = useState(true);
	const [gpuStatus, setGpuStatus] = useState({ temp: 0, load: 0 });

	useEffect(() => {
		// Simulate GPU status updates
		const interval = setInterval(() => {
			setGpuStatus({
				temp: Math.floor(Math.random() * 30) + 50, // 50-80°C
				load: Math.floor(Math.random() * 60) + 40, // 40-100%
			});
		}, 2000);

		return () => clearInterval(interval);
	}, []);

	const handleSubmit = (e: React.FormEvent) => {
		e.preventDefault();
		if (!input.trim() || isStreaming) return;

		const userMessage = { role: "user", content: input };
		setMessages((prev) => [...prev, userMessage]);
		setInput("");
		makeModelCalls(input);
	};

	const makeModelCalls = async (prompt: string) => {
		setIsStreaming(true);

		// Add new assistant messages while preserving all previous messages
		setMessages((prev) => [
			...prev,
			{ role: "assistant", content: "", isStreaming: true },
			{ role: "assistant-fast", content: "", isStreaming: true },
		]);

		// Start both streams in parallel - use distinct variables to avoid race conditions
		const standardPromise = streamFromServer(
			prompt,
			"assistant",
			"claude-3-7-sonnet-20250219",
		);
		const fastPromise = streamFromServer(prompt, "assistant-fast", "deepseek");

		Promise.all([standardPromise, fastPromise])
			.then(() => {
				// Only set isStreaming to false when both are complete
				setIsStreaming(false);
			})
			.catch((error) => {
				console.error("Error in streaming:", error);
				setIsStreaming(false);
			});
	};

	const streamFromServer = async (
		prompt: string,
		role: string,
		model: string,
	) => {
		try {
			console.log(`Starting stream for ${role} using model ${model}`);
			const response = await fetch("/api/chat", {
				method: "POST",
				headers: {
					"Content-Type": "application/json",
				},
				body: JSON.stringify({
					prompt,
					model,
				}),
			});

			if (!response.ok) {
				throw new Error(`Server responded with ${response.status}`);
			}

			const reader = response.body?.getReader();
			if (!reader) {
				throw new Error("Response body is null");
			}

			const decoder = new TextDecoder();
			let fullResponse = "";

			while (true) {
				const { done, value } = await reader.read();

				if (done) {
					console.log(`Stream complete for ${role}`);
					break;
				}

				// Process the chunks as Server-Sent Events
				const chunk = decoder.decode(value);
				const lines = chunk.split("\n\n");

				for (const line of lines) {
					if (line.startsWith("data: ")) {
						const data = line.substring(6);

						if (data === "[DONE]") {
							continue;
						}

						try {
							const parsed = JSON.parse(data);
							if (parsed.content) {
								fullResponse += parsed.content;
								console.log(
									`Received content for ${role}: ${parsed.content.substring(0, 20)}...`,
								);

								setMessages((prev) => {
									// Find the most recent message with this role that's streaming
									const lastIndex = [...prev].reverse().findIndex(
										(msg) => msg.role === role && msg.isStreaming
									);
									const mostRecentIndex = lastIndex !== -1 ? prev.length - 1 - lastIndex : -1;
									
									if (mostRecentIndex !== -1) {
										return prev.map((msg, idx) => {
											if (idx === mostRecentIndex) {
												return {
													...msg,
													content: fullResponse,
													isStreaming: true,
												};
											}
											return msg;
										});
									}
									return prev;
								});
							}
						} catch (e) {
							console.error("Error parsing SSE data:", e);
						}
					}
				}
			}

			// Mark streaming as complete for this message
			setMessages((prev) => {
				// Find the most recent message with this role that's streaming
				const lastIndex = [...prev].reverse().findIndex(
					(msg) => msg.role === role && msg.isStreaming
				);
				const mostRecentIndex = lastIndex !== -1 ? prev.length - 1 - lastIndex : -1;
				
				if (mostRecentIndex !== -1) {
					return prev.map((msg, idx) => {
						if (idx === mostRecentIndex) {
							return { ...msg, isStreaming: false };
						}
						return msg;
					});
				}
				return prev;
			});

			console.log(
				`Stream finalized for ${role}, message length: ${fullResponse.length}`,
			);
		} catch (error) {
			console.error(`Error with ${model} streaming:`, error);

			// Handle error by finishing the stream with an error message
			setMessages((prev) => {
				// Find the most recent message with this role that's streaming
				const lastIndex = [...prev].reverse().findIndex(
					(msg) => msg.role === role && msg.isStreaming
				);
				const mostRecentIndex = lastIndex !== -1 ? prev.length - 1 - lastIndex : -1;
				
				if (mostRecentIndex !== -1) {
					return prev.map((msg, idx) => {
						if (idx === mostRecentIndex) {
							return {
								...msg,
								content: `${msg.content || ""}\n\n[ERROR] Failed to complete response. API error occurred.`,
								isStreaming: false,
							};
						}
						return msg;
					});
				}
				return prev;
			});
		}

		// Return a resolved promise for Promise.all to work with
		return Promise.resolve();
	};

	return (
		<main className="flex min-h-screen flex-col bg-black text-green-500 font-mono">
			{/* {showModal && <GPUModal onClose={() => setShowModal(false)} />} */}
			<div className="flex flex-col h-screen">
				<header className="border-b border-green-900/30 p-4">
					<h1 className="text-xl font-bold text-center tracking-tight">
						KernelForge GPU CLUSTER v0.0.5
					</h1>
				</header>

				{/* <StatusBar gpuTemp={gpuStatus.temp} gpuLoad={gpuStatus.load} /> */}

				<div className="flex flex-1 overflow-hidden">
					<ChatWindow
						title="STANDARD::MODEL"
						description="Claude-3-7-Sonnet-Thinking (Standard)"
						messages={messages.filter((m) => m.role !== "assistant-fast")}
						className="border-r border-green-900/30 w-1/2"
						isStreaming={isStreaming}
					/>

					<ChatWindow
						title="SPECULATIVE::MODEL"
						description="DeepSeek-R1-Distill-32B (Fast)"
						messages={messages.filter((m) => m.role !== "assistant")}
						className="w-1/2"
						isSpeculative={true}
						isStreaming={isStreaming}
					/>
				</div>

				<footer className="border-t border-green-900/30 p-4">
					<form
						onSubmit={handleSubmit}
						className="flex gap-2 max-w-3xl mx-auto"
					>
						<Input
							value={input}
							onChange={(e) => setInput(e.target.value)}
							placeholder="$ execute_command --gpu"
							className="bg-black border-green-900/50 text-green-500 focus-visible:ring-green-900/50 font-mono"
							disabled={isStreaming}
						/>
						<Button
							type="submit"
							disabled={isStreaming || !input.trim()}
							className="bg-green-900/30 hover:bg-green-900/50 text-green-500 border border-green-900/50"
						>
							<Send className="h-4 w-4" />
							<span className="sr-only">Send</span>
						</Button>
					</form>
					<p className="text-xs text-green-700 text-center mt-2 font-mono">
						[SYS]: Command will be executed using parallel LLM processing
					</p>
				</footer>
			</div>
		</main>
	);
}
