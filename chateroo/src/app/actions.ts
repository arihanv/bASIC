'use server';

import { openai } from "./openai";

export async function getResponse(prompt: string) {
    const stream=await openai.chat.completions.create({
        model: 'gpt-4o-mini',
        messages: [
            {
                role: 'developer', //developer says you should do this, user asks for something, and assistant gives you examples
                content: [
                    {
                        type: 'text',
                        text: 'You are a condescending VIP, kind of like Larry Summers'
                    }
                ]
            },
            {
                role: 'user',
                content: [
                    {
                        type: 'text',
                        text: 'Can you please help me open the door?'
                    }
                ]
            }
        ]
    });
    return stream;
    // return 'hi';
}

