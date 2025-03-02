# Dual Chat Streaming with OpenAI

This application demonstrates real-time streaming from OpenAI models, comparing a standard model (o1) with a faster model (o1-mini) side by side.

## Setup

1. Clone this repository
2. Install dependencies:
   ```
   npm install
   ```
   or
   ```
   bun install
   ```

3. Create a `.env.local` file in the root directory with your OpenAI API key:
   ```
   NEXT_PUBLIC_OPENAI_API_KEY=your_openai_api_key_here
   ```

4. Start the development server:
   ```
   npm run dev
   ```
   or
   ```
   bun dev
   ```

5. Open [http://localhost:3000](http://localhost:3000) in your browser

## Features

- Real-time streaming from two different OpenAI models
- Side-by-side comparison of model outputs
- Futuristic "quantum GPU" themed UI
- Simulated GPU temperature and load indicators

## Technologies Used

- Next.js
- React
- OpenAI API
- TypeScript
- Tailwind CSS
- shadcn/ui components

## API Key Security Note

This application uses `NEXT_PUBLIC_` prefixed environment variables which makes the OpenAI API key available on the client side. This is for demonstration purposes only and should not be used in production.

For production applications:
- Use server-side API routes to make API calls
- Store API keys in server-side environment variables only
- Implement proper authentication and rate limiting 