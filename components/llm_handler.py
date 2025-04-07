"""
LLM handler component for Jarvis AI Assistant.

Handles interactions with the local language model.
"""

import os
import logging
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, set_seed
from utils import chunked_text

logger = logging.getLogger(__name__)

class LLMHandler:
    """Handles interactions with the local language model."""
    
    def __init__(self, config):
        """
        Initialize the LLM handler.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.llm_config = config.get_llm_config()
        self.app_config = config.get_app_config()
        
        self.model_name = self.llm_config["model_name"]
        self.model_path = self.llm_config["model_path"]
        self.use_gpu = self.llm_config["use_gpu"] and torch.cuda.is_available()
        self.max_new_tokens = self.llm_config["max_new_tokens"]
        self.temperature = self.llm_config["temperature"]
        
        # Set device
        self.device = "cuda" if self.use_gpu else "cpu"
        
        # Initialize tokenizer and model
        self.tokenizer = None
        self.model = None
        self.pipe = None
        
        # Set random seed for reproducibility
        set_seed(42)
        
        # System prompt components
        self.assistant_name = self.app_config["name"]
        self.assistant_prompt = self.app_config["assistant_prompt"]
    
    def _load_model(self):
        """Load the language model and tokenizer."""
        if self.model is not None:
            return
        
        logger.info(f"Loading language model: {self.model_name}")
        
        try:
            # Try to load from local path first
            if os.path.exists(self.model_path):
                logger.info(f"Loading model from local path: {self.model_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16 if self.use_gpu else torch.float32,
                    device_map="auto" if self.use_gpu else None,
                    low_cpu_mem_usage=True
                )
            else:
                # Fall back to downloading from Hugging Face
                logger.info(f"Loading model from Hugging Face: {self.model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.use_gpu else torch.float32,
                    device_map="auto" if self.use_gpu else None,
                    low_cpu_mem_usage=True
                )
            
            # Create text generation pipeline
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.use_gpu else -1
            )
            
            logger.info(f"Model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Error loading language model: {e}")
            raise RuntimeError(f"Failed to load language model: {e}")
    
    def get_system_prompt(self):
        """
        Get the system prompt for the assistant.
        
        Returns:
            str: System prompt
        """
        return f"""You are {self.assistant_name}, an AI assistant running completely offline on the user's machine.
{self.assistant_prompt}

Current date and time: {time.strftime("%Y-%m-%d %H:%M:%S")}

Keep your responses concise and helpful. If you can't perform a task because you're running offline,
suggest alternatives that might work locally. When handling commands, clearly indicate what you plan to do.
"""
    
    def format_prompt(self, user_input, conversation_history=None):
        """
        Format the prompt for the language model.
        
        Args:
            user_input (str): User's input
            conversation_history (list, optional): Conversation history
            
        Returns:
            str: Formatted prompt
        """
        system_prompt = self.get_system_prompt()
        
        # Format conversation with system prompt
        formatted_prompt = system_prompt + "\n\n"
        
        # Add conversation history if provided
        if conversation_history and len(conversation_history) > 0:
            for entry in conversation_history:
                if entry["role"] == "user":
                    formatted_prompt += f"User: {entry['content']}\n"
                else:  # assistant
                    formatted_prompt += f"{self.assistant_name}: {entry['content']}\n"
        
        # Add current user input
        formatted_prompt += f"User: {user_input}\n{self.assistant_name}:"
        
        return formatted_prompt
    
    def generate_response(self, user_input, conversation_history=None):
        """
        Generate a response to the user's input.
        
        Args:
            user_input (str): User's input
            conversation_history (list, optional): Conversation history
            
        Returns:
            str: Generated response
        """
        # Load model if not loaded
        if self.model is None:
            self._load_model()
        
        # Format the prompt
        prompt = self.format_prompt(user_input, conversation_history)
        
        try:
            start_time = time.time()
            
            # Generate response
            outputs = self.pipe(
                prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=0.9,
                top_k=50,
                do_sample=True,
                num_return_sequences=1,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Extract generated text
            generated_text = outputs[0]["generated_text"]
            
            # Extract only the assistant's response (remove the prompt)
            assistant_response = generated_text[len(prompt):].strip()
            
            # Clean up the response (remove any additional "User:" or similar)
            if "User:" in assistant_response:
                assistant_response = assistant_response.split("User:")[0].strip()
            
            logger.info(f"Response generated in {time.time() - start_time:.2f} seconds")
            
            return assistant_response
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I'm sorry, I encountered an error while processing your request."
    
    def summarize_text(self, text):
        """
        Summarize long text.
        
        Args:
            text (str): Text to summarize
            
        Returns:
            str: Summarized text
        """
        # Load model if not loaded
        if self.model is None:
            self._load_model()
        
        # For very long texts, chunk and summarize each part
        if len(text) > 8000:
            chunks = chunked_text(text, chunk_size=8000)
            summaries = []
            
            for i, chunk in enumerate(chunks):
                prompt = f"""You are {self.assistant_name}, an AI assistant. Summarize the following text (part {i+1} of {len(chunks)}) concisely:

Text:
{chunk}

Summary:"""
                
                try:
                    output = self.pipe(
                        prompt,
                        max_new_tokens=min(300, self.max_new_tokens),
                        temperature=0.3,  # Lower temperature for more deterministic summary
                        top_p=0.9,
                        do_sample=True,
                        num_return_sequences=1
                    )
                    
                    generated_text = output[0]["generated_text"]
                    summary = generated_text.split("Summary:")[1].strip() if "Summary:" in generated_text else generated_text[len(prompt):].strip()
                    summaries.append(summary)
                except Exception as e:
                    logger.error(f"Error summarizing chunk {i+1}: {e}")
                    summaries.append(f"[Error summarizing part {i+1}]")
            
            # Combine chunk summaries
            if len(summaries) > 1:
                combined_summary = "\n\n".join(summaries)
                
                # Create a final summary of the chunk summaries
                final_summary_prompt = f"""You are {self.assistant_name}, an AI assistant. Create a final coherent summary from these section summaries:

Section Summaries:
{combined_summary}

Final Summary:"""
                
                try:
                    output = self.pipe(
                        final_summary_prompt,
                        max_new_tokens=min(500, self.max_new_tokens),
                        temperature=0.3,
                        top_p=0.9,
                        do_sample=True,
                        num_return_sequences=1
                    )
                    
                    generated_text = output[0]["generated_text"]
                    final_summary = generated_text.split("Final Summary:")[1].strip() if "Final Summary:" in generated_text else generated_text[len(final_summary_prompt):].strip()
                    return final_summary
                except Exception as e:
                    logger.error(f"Error creating final summary: {e}")
                    return combined_summary
            else:
                return summaries[0]
        else:
            # For shorter texts, summarize directly
            prompt = f"""You are {self.assistant_name}, an AI assistant. Summarize the following text concisely:

Text:
{text}

Summary:"""
            
            try:
                output = self.pipe(
                    prompt,
                    max_new_tokens=min(300, self.max_new_tokens),
                    temperature=0.3,
                    top_p=0.9,
                    do_sample=True,
                    num_return_sequences=1
                )
                
                generated_text = output[0]["generated_text"]
                summary = generated_text.split("Summary:")[1].strip() if "Summary:" in generated_text else generated_text[len(prompt):].strip()
                return summary
            except Exception as e:
                logger.error(f"Error summarizing text: {e}")
                return "I encountered an error while trying to summarize the text."
    
    def analyze_intent(self, user_input):
        """
        Analyze the user's intent from their input.
        
        Args:
            user_input (str): User's input
            
        Returns:
            dict: Intent analysis
        """
        # Load model if not loaded
        if self.model is None:
            self._load_model()
            
        prompt = f"""You are {self.assistant_name}, an AI assistant. Analyze the intent of the following user input:

Input: "{user_input}"

Respond with a JSON object containing:
1. "intent": The main intent category (one of: "question", "command", "conversation", "information_request")
2. "action": The specific action to take if any
3. "entities": Key entities mentioned in the input
4. "confidence": Your confidence in this analysis (0.0 to 1.0)

JSON:"""
        
        try:
            output = self.pipe(
                prompt,
                max_new_tokens=200,
                temperature=0.1,  # Low temperature for more deterministic output
                top_p=0.9,
                do_sample=False,
                num_return_sequences=1
            )
            
            generated_text = output[0]["generated_text"]
            
            # Extract JSON part
            json_str = generated_text.split("JSON:")[1].strip() if "JSON:" in generated_text else generated_text[len(prompt):].strip()
            
            # Clean up the string to make it valid JSON
            import json
            import re
            
            # Remove markdown code block markers if present
            json_str = re.sub(r'```json|```', '', json_str).strip()
            
            # Parse JSON
            try:
                intent_data = json.loads(json_str)
                return intent_data
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse intent JSON: {json_str}")
                # Return a default fallback intent
                return {
                    "intent": "conversation",
                    "action": None,
                    "entities": [],
                    "confidence": 0.5
                }
        except Exception as e:
            logger.error(f"Error analyzing intent: {e}")
            return {
                "intent": "conversation",
                "action": None,
                "entities": [],
                "confidence": 0.3
            }
    
    def update_model_parameters(self, temperature=None, max_new_tokens=None):
        """
        Update model generation parameters.
        
        Args:
            temperature (float, optional): Generation temperature
            max_new_tokens (int, optional): Maximum new tokens to generate
            
        Returns:
            bool: True if successful, False otherwise
        """
        if temperature is not None:
            self.temperature = max(0.1, min(1.5, temperature))  # Clamp between 0.1 and 1.5
            self.config.update("llm", "temperature", self.temperature)
        
        if max_new_tokens is not None:
            self.max_new_tokens = max(50, min(2000, max_new_tokens))  # Clamp between 50 and 2000
            self.config.update("llm", "max_new_tokens", self.max_new_tokens)
        
        return True
    
    def cleanup(self):
        """Clean up resources."""
        # Free up GPU memory
        if self.model is not None and self.use_gpu:
            self.model = None
            self.tokenizer = None
            self.pipe = None
            
            # Force garbage collection
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
