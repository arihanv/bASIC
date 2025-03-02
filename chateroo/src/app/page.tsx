'use client';

import { CSSProperties, useState } from 'react';
import Markdown from 'react-markdown';
import { IconArrowUp } from '@tabler/icons-react';

export type Message={
  saidBy:
  | 'user' //user said
  | 'assistant' //previous messages
  content: string;
};

const textboxStyle:{[key: string]: CSSProperties}={
    both: {
        width: 'fit-content',
        maxWidth: 800,
        fontWeight: 'bold',
        borderRadius: 12,
        padding: '5px 10px',
    },
    user: {
        backgroundColor: '#E9E9EB',
        alignSelf: 'flex-end'
    },
    assistant: {
        backgroundColor: '#1789FE',
        color: 'white',
    }
};

export default function Home() {
    const [messages, setMessages]=useState<Message[]>([]);
    const [userPrompt, setUserPrompt]=useState('');

    function sendMessage() {
        const thePrompt=userPrompt;
        setUserPrompt('');
        setMessages(messages=>[...messages, {
            saidBy: 'user',
            content: thePrompt,
        }, {
            saidBy: 'assistant',
            content: ''
        }]);
        fetch('/api/gpt', {
            method: 'POST',
            body: JSON.stringify({
                query: thePrompt
            })
        })
        .then(async res=>{
            if (!res.body) return;
            const decoder=new TextDecoder();
            const reader=res.body.getReader();
            while (true) {
                const { done, value }=await reader.read();
                const chunk=decoder.decode(value);
                if (done) {
                    break;
                }
                setMessages(messages=>{
                    const lastMessage=messages.at(-1)!;
                    lastMessage.content=lastMessage.content+chunk;
                    return messages.slice(0, -1).concat([lastMessage]);
                });
            }
        });
    }
    
    return <div>
        <h1 className='font-bold text-center my-3 text-2xl'>Chateroo</h1>
        <div className="flex flex-col px-5">
            {
                messages.map((message, i)=>{
                    return <div style={{...textboxStyle.both, ...textboxStyle[message.saidBy]}} key={i}>
                        <Markdown>{message.content || '...'}</Markdown>
                    </div>;
                })
            }
        </div>

        <div className="absolute bottom-10 left-[10vw] w-[80vw]">
            <div
                className='py-1 px-2 m-3 flex'
                style={{
                    border: '1px solid black',
                    width: '100%',
                    borderRadius: 8
                }}
            >
                <input
                    value={userPrompt}
                    autoFocus
                    placeholder='Enter your message...'
                    style={{
                        outline: 'none',
                        width: '100%'
                    }}
                    onKeyDown={async e=>{
                        if (e.key==='Enter') {
                            sendMessage();
                        }
                    }}
                    onChange={e=>{
                        setUserPrompt(e.target.value);
                    }}
                />
                <button className='bg-black rounded-md w-7 h-7 grid place-items-center hover:bg-gray-700 cursor-pointer' onClick={sendMessage}>
                    <IconArrowUp color='white' className='relative left-[.5px]' />
                </button>
            </div>
        </div>
    </div>;
}
