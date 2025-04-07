"""
Memory component for Jarvis AI Assistant.

Handles vector storage for natural language memory and conversation history.
"""

import os
import logging
import json
import time
import uuid
from datetime import datetime
from pathlib import Path
import threading
import torch

# Use sentence-transformers for embeddings
from sentence_transformers import SentenceTransformer

# Use ChromaDB for vector storage
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)

class Memory:
    """Handles natural language memory and conversation history."""
    
    def __init__(self, config):
        """
        Initialize the memory component.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.memory_config = config.get_memory_config()
        
        self.vector_db_path = self.memory_config["vector_db_path"]
        self.history_size = self.memory_config["history_size"]
        self.embedding_model_name = self.memory_config["embedding_model"]
        
        # Create vector DB directory if it doesn't exist
        Path(self.vector_db_path).mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.vector_db_path,
            settings=Settings(
                anonymized_telemetry=False
            )
        )
        
        # Get or create collections
        self.memory_collection = self._get_or_create_collection("long_term_memory")
        self.conversation_collection = self._get_or_create_collection("conversation_history")
        
        # Initialize sentence transformer for embeddings
        self.embedding_model = None
        self.embedding_fn = None
        
        # Initialize embedding function
        self._init_embedding_function()
        
        # Conversation history (in-memory cache)
        self.conversation_history = []
        
        # Load conversation history from DB
        self._load_conversation_history()
        
        # Lock for thread safety
        self.lock = threading.RLock()
    
    def _init_embedding_function(self):
        """Initialize the embedding function."""
        try:
            # If CUDA is available, use it for the embedding model
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # For sentence-transformers
            logger.info(f"Loading embedding model {self.embedding_model_name} on {device}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name, device=device)
            
            # Define embedding function for ChromaDB
            self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.embedding_model_name,
                device=device
            )
            
            # Update collections with embedding function
            self.memory_collection = self.client.get_collection(
                name="long_term_memory",
                embedding_function=self.embedding_fn
            )
            
            self.conversation_collection = self.client.get_collection(
                name="conversation_history",
                embedding_function=self.embedding_fn
            )
            
            logger.info("Embedding function initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing embedding function: {e}")
            # Fall back to default embedding function
            logger.info("Falling back to default embedding function")
            self.embedding_fn = None
    
    def _get_or_create_collection(self, name):
        """
        Get or create a ChromaDB collection.
        
        Args:
            name (str): Collection name
            
        Returns:
            Collection: ChromaDB collection
        """
        try:
            # Try to get existing collection
            return self.client.get_collection(name=name)
        except Exception:
            # Create new collection if it doesn't exist
            logger.info(f"Creating new collection: {name}")
            return self.client.create_collection(name=name)
    
    def _load_conversation_history(self):
        """Load conversation history from DB."""
        try:
            # Get all items from conversation collection
            results = self.conversation_collection.get()
            
            if results and results['ids']:
                # Sort by timestamp
                entries = []
                for i, item_id in enumerate(results['ids']):
                    metadata = results['metadatas'][i]
                    documents = results['documents'][i]
                    
                    timestamp = metadata.get('timestamp', 0)
                    entries.append({
                        'id': item_id,
                        'timestamp': timestamp,
                        'role': metadata.get('role', 'unknown'),
                        'content': documents
                    })
                
                # Sort by timestamp
                entries.sort(key=lambda x: x['timestamp'])
                
                # Take only the last history_size entries
                entries = entries[-self.history_size:]
                
                # Convert to conversation history format
                self.conversation_history = [
                    {'role': entry['role'], 'content': entry['content']}
                    for entry in entries
                ]
                
                logger.info(f"Loaded {len(self.conversation_history)} conversation history entries")
        except Exception as e:
            logger.error(f"Error loading conversation history: {e}")
            self.conversation_history = []
    
    def add_to_conversation(self, role, content):
        """
        Add an entry to the conversation history.
        
        Args:
            role (str): Role ('user' or 'assistant')
            content (str): Content of the message
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not content or not role:
            return False
        
        with self.lock:
            try:
                # Add to in-memory cache
                entry = {'role': role, 'content': content}
                self.conversation_history.append(entry)
                
                # Limit history size
                if len(self.conversation_history) > self.history_size:
                    self.conversation_history = self.conversation_history[-self.history_size:]
                
                # Add to vector DB
                timestamp = int(time.time())
                entry_id = str(uuid.uuid4())
                
                self.conversation_collection.add(
                    ids=[entry_id],
                    documents=[content],
                    metadatas=[{
                        'role': role,
                        'timestamp': timestamp,
                        'date': datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
                    }]
                )
                
                return True
            except Exception as e:
                logger.error(f"Error adding to conversation history: {e}")
                return False
    
    def get_conversation_history(self, limit=None):
        """
        Get the conversation history.
        
        Args:
            limit (int, optional): Maximum number of entries to return
            
        Returns:
            list: Conversation history
        """
        with self.lock:
            if limit is None or limit >= len(self.conversation_history):
                return self.conversation_history
            else:
                return self.conversation_history[-limit:]
    
    def clear_conversation_history(self):
        """
        Clear the conversation history.
        
        Returns:
            bool: True if successful, False otherwise
        """
        with self.lock:
            try:
                # Clear in-memory cache
                self.conversation_history = []
                
                # Clear vector DB collection
                self.conversation_collection.delete(where={})
                
                logger.info("Conversation history cleared")
                return True
            except Exception as e:
                logger.error(f"Error clearing conversation history: {e}")
                return False
    
    def add_memory(self, content, metadata=None):
        """
        Add an entry to long-term memory.
        
        Args:
            content (str): Content to remember
            metadata (dict, optional): Additional metadata
            
        Returns:
            str: Memory ID if successful, None otherwise
        """
        if not content:
            return None
        
        with self.lock:
            try:
                # Create metadata if not provided
                if metadata is None:
                    metadata = {}
                
                # Add timestamp
                timestamp = int(time.time())
                metadata['timestamp'] = timestamp
                metadata['date'] = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
                
                # Generate ID
                memory_id = str(uuid.uuid4())
                
                # Add to vector DB
                self.memory_collection.add(
                    ids=[memory_id],
                    documents=[content],
                    metadatas=[metadata]
                )
                
                logger.info(f"Added memory: {memory_id}")
                return memory_id
            except Exception as e:
                logger.error(f"Error adding memory: {e}")
                return None
    
    def search_memory(self, query, limit=5):
        """
        Search long-term memory.
        
        Args:
            query (str): Search query
            limit (int, optional): Maximum number of results
            
        Returns:
            list: Search results
        """
        if not query:
            return []
        
        with self.lock:
            try:
                results = self.memory_collection.query(
                    query_texts=[query],
                    n_results=limit
                )
                
                # Format results
                formatted_results = []
                for i, item_id in enumerate(results['ids'][0]):
                    formatted_results.append({
                        'id': item_id,
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i] if 'metadatas' in results and results['metadatas'] else {}
                    })
                
                return formatted_results
            except Exception as e:
                logger.error(f"Error searching memory: {e}")
                return []
    
    def delete_memory(self, memory_id):
        """
        Delete a memory entry.
        
        Args:
            memory_id (str): Memory ID to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not memory_id:
            return False
        
        with self.lock:
            try:
                self.memory_collection.delete(ids=[memory_id])
                logger.info(f"Deleted memory: {memory_id}")
                return True
            except Exception as e:
                logger.error(f"Error deleting memory: {e}")
                return False
    
    def get_all_memories(self, limit=100, offset=0):
        """
        Get all memories with pagination.
        
        Args:
            limit (int, optional): Maximum number of results
            offset (int, optional): Offset for pagination
            
        Returns:
            list: Memory entries
        """
        with self.lock:
            try:
                # Get all memories
                results = self.memory_collection.get()
                
                if not results or not results['ids']:
                    return []
                
                # Format results
                all_memories = []
                for i, item_id in enumerate(results['ids']):
                    all_memories.append({
                        'id': item_id,
                        'content': results['documents'][i],
                        'metadata': results['metadatas'][i] if 'metadatas' in results and results['metadatas'] else {}
                    })
                
                # Sort by timestamp (newest first)
                all_memories.sort(key=lambda x: x['metadata'].get('timestamp', 0), reverse=True)
                
                # Apply pagination
                paginated_memories = all_memories[offset:offset + limit]
                
                return paginated_memories
            except Exception as e:
                logger.error(f"Error getting all memories: {e}")
                return []
    
    def get_memory_by_id(self, memory_id):
        """
        Get a memory by ID.
        
        Args:
            memory_id (str): Memory ID
            
        Returns:
            dict: Memory entry or None if not found
        """
        if not memory_id:
            return None
        
        with self.lock:
            try:
                result = self.memory_collection.get(ids=[memory_id])
                
                if not result or not result['ids']:
                    return None
                
                return {
                    'id': result['ids'][0],
                    'content': result['documents'][0],
                    'metadata': result['metadatas'][0] if 'metadatas' in result and result['metadatas'] else {}
                }
            except Exception as e:
                logger.error(f"Error getting memory by ID: {e}")
                return None
    
    def export_memories(self, file_path):
        """
        Export all memories to a JSON file.
        
        Args:
            file_path (str): Path to save the export
            
        Returns:
            bool: True if successful, False otherwise
        """
        with self.lock:
            try:
                memories = self.get_all_memories(limit=10000)  # Get all memories
                
                with open(file_path, 'w') as f:
                    json.dump(memories, f, indent=2)
                
                logger.info(f"Exported {len(memories)} memories to {file_path}")
                return True
            except Exception as e:
                logger.error(f"Error exporting memories: {e}")
                return False
    
    def import_memories(self, file_path):
        """
        Import memories from a JSON file.
        
        Args:
            file_path (str): Path to the import file
            
        Returns:
            int: Number of imported memories
        """
        with self.lock:
            try:
                with open(file_path, 'r') as f:
                    memories = json.load(f)
                
                imported_count = 0
                
                for memory in memories:
                    content = memory.get('content')
                    metadata = memory.get('metadata', {})
                    
                    if content:
                        self.add_memory(content, metadata)
                        imported_count += 1
                
                logger.info(f"Imported {imported_count} memories from {file_path}")
                return imported_count
            except Exception as e:
                logger.error(f"Error importing memories: {e}")
                return 0
    
    def cleanup(self):
        """Clean up resources."""
        # No specific cleanup needed
        pass
