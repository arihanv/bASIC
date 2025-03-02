import { openai } from "@/app/openai";
import { NextRequest } from "next/server";

export async function POST(req: NextRequest) {
    const { query }=await req.json();
    const response=await openai.chat.completions.create({
        model: 'gpt-4o-mini',
        messages: [
            {
                role: 'user',
                content: query
            }
        ],
        stream: true
    });
    const encoder=new TextEncoder();
    // for await (const chunk of response) {
    //     console.log('chunk', chunk.choices[0].delta.content);
    // }
    const stream=new ReadableStream({
        async start(controller) {
            for await (const chunk of response) { // Process the GPT stream
                const content=chunk.choices[0].delta.content || '';
                if (content) {
                    controller.enqueue(encoder.encode(content));
                }
            }
            // Sets done to true
            controller.close();
        }
    })
    return new Response(stream);
}

