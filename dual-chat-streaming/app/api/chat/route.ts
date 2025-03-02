import { NextRequest, NextResponse } from 'next/server';
import OpenAI from 'openai';

const openai = new OpenAI({
  apiKey: process.env.ANTHROPIC_API_KEY,
  baseURL: "https://api.anthropic.com/v1/",
});

const deepseek = new OpenAI({
    baseURL: "https://api--openai-vllm--d8zwcx9rqzwl.code.run/v1",
    apiKey: "EMPTY"
});

export async function POST(req: NextRequest) {
  try {
    const { prompt, model } = await req.json();

    if (!prompt) {
      return NextResponse.json(
        { error: "Prompt is required" },
        { status: 400 }
      );
    }

    let response: any;

    console.log(`Processing request for model: ${model}`);

    if (model === "deepseek") {
        response = await deepseek.chat.completions.create({
            model: "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            max_tokens: 8000,
            messages: [{ role: "user", content: prompt }],
            stream: true,
        });
        console.log("DeepSeek request initiated");
    } else {
        // Use a try-catch here since the Anthropic API might have custom parameters
        try {
            response = await openai.chat.completions.create({
                model: model,
                max_tokens: 4000,
                messages: [{ role: "user", content: prompt }],
                stream: true,
                // Custom Anthropic params - handle separately to avoid TypeScript errors
                ...(model.includes("claude") ? {
                    thinking: {
                        type: "enabled",
                        budget_tokens: 2000
                    },
                    betas: ["output-128k-2025-02-19"],
                } : {})
            } as any);
            console.log("Claude request initiated");
        } catch (error) {
            console.error("Error initiating Claude request:", error);
            throw error;
        }
    }

    // Create a transform stream
    const encoder = new TextEncoder();
    const stream = new ReadableStream({
      async start(controller) {
        try {
          // Handle each chunk
          for await (const chunk of response) {
            // Log the first few chunks for debugging
            // console.log("Raw chunk:", JSON.stringify(chunk).substring(0, 200) + "...");
              
            // Skip null/undefined chunks
            if (chunk === null || chunk === undefined) {
              continue;
            }
            
            // Handle Anthropic ping messages
            if (chunk.type === 'ping') {
              continue;
            }
            
            try {
              // Extract content from either Anthropic or OpenAI format
              let text = '';
              
              // Check if it's a valid chunk with choices
              if (chunk.choices && chunk.choices.length > 0) {
                // Handle different formats of the responses
                if (chunk.choices[0]?.delta?.content !== undefined) {
                  text = chunk.choices[0].delta.content;
                } else if (chunk.choices[0]?.delta?.text !== undefined) {
                  text = chunk.choices[0].delta.text;
                } else if (chunk.choices[0]?.text !== undefined) {
                  text = chunk.choices[0].text;
                } else if (chunk.choices[0]?.content !== undefined) {
                  text = chunk.choices[0].content;
                }
              }
              
              if (text) {
                controller.enqueue(encoder.encode(`data: ${JSON.stringify({ content: text })}\n\n`));
              }
            } catch (e) {
              console.error('Error processing chunk:', e, 'Chunk:', JSON.stringify(chunk));
            }
          }
        } catch (error) {
          console.error("Error in stream processing:", error);
        }
        
        controller.enqueue(encoder.encode('data: [DONE]\n\n'));
        controller.close();
      }
    });

    return new Response(stream, {
      headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
      },
    });
  } catch (error) {
    console.error('Error in chat route:', error);
    return NextResponse.json(
      { error: "Failed to generate response" },
      { status: 500 }
    );
  }
} 