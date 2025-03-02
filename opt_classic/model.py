import time
import torch
import requests
from transformers import AutoTokenizer
from openai import OpenAI
from typing import List, Dict, Union, Any, Optional

class SPModel(torch.nn.Module):
    def __init__(
            self,
            base_model_name_or_path,
            draft_model_name_or_path=None,
            api_base_url="https://api--openai-vllm--d8zwcx9rqzwl.code.run/v1",
            api_key="EMPTY"
    ):
        super().__init__()
        self.base_model_name_or_path = base_model_name_or_path
        self.draft_model_name_or_path = draft_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
        
        # Initialize OpenAI client
        self.client = OpenAI(
            base_url=api_base_url,
            api_key=api_key
        )
        
        # For compatibility with existing code
        self.config = type('obj', (object,), {
            'vocab_size': self.tokenizer.vocab_size
        })
        
    def get_tokenizer(self):
        return self.tokenizer
        
    @classmethod
    def from_pretrained(
            cls,
            base_model_path=None,
            draft_model_path=None,
            api_base_url="https://api--openai-vllm--d8zwcx9rqzwl.code.run/v1",
            api_key="EMPTY",
            **kwargs
    ):
        return cls(
            base_model_name_or_path=base_model_path,
            draft_model_name_or_path=draft_model_path,
            api_base_url=api_base_url,
            api_key=api_key
        )
    
    def eval(self):
        # No-op for compatibility
        return self
        
    @torch.no_grad()
    def spgenerate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            nodes=50,
            threshold=0.5,
            max_depth=10,
            print_time=False,
            speculative_model=None
    ):
        # Convert input_ids to text
        prompt = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        
        # Set up parameters
        if temperature < 1e-5:
            temperature = 0.0
            
        # Define speculative model if provided
        extra_body = {}
        if self.draft_model_name_or_path and self.draft_model_name_or_path != self.base_model_name_or_path:
            extra_body["speculative_model"] = self.draft_model_name_or_path
        
        start_time = time.time()
        
        # Call the API with streaming
        try:
            completion = self.client.chat.completions.create(
                model=self.base_model_name_or_path,
                stream=True,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_new_tokens,
                top_p=top_p if top_p > 0 else None,
                extra_body=extra_body if extra_body else None
            )
            
            # Collect the generated text
            generated_text = ""
            for chunk in completion:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    generated_text += content
                    if print_time:
                        print(content, end="", flush=True)
                        
            end_time = time.time()
            if print_time:
                print(f"\nGeneration time: {end_time - start_time:.2f} seconds")
                
            # Create output tensor
            full_text = prompt + generated_text
            output_ids = self.tokenizer(full_text, return_tensors="pt").input_ids
            
            print(f"Generated {len(generated_text)} characters")
            return output_ids
            
        except Exception as e:
            print(f"Error in API call: {e}")
            # Return input as fallback
            return input_ids