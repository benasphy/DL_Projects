import numpy as np
import os
import tensorflow as tf

# Configure TensorFlow to handle GPU properly
try:
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print(f"Metal GPU enabled: {physical_devices}")
        
        # Note: Mixed precision is disabled due to compatibility issues with CudnnRNN
        # tf.keras.mixed_precision.set_global_policy('mixed_float16')
except Exception as e:
    print(f"GPU configuration error: {e}")
    print("Using CPU instead")
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

class ImprovedTextGenerator:
    """
    A class to improve text generation quality using various techniques
    """
    
    def __init__(self, model, tokenizer, max_sequence_length, char_mode=False, 
                 char_to_idx=None, idx_to_char=None, vocab_size=None):
        """
        Initialize the improved text generator
        
        Args:
            model: The trained model
            tokenizer: The tokenizer used for word-level generation
            max_sequence_length: Maximum sequence length
            char_mode: Whether the model is character-level
            char_to_idx: Character to index mapping (for char mode)
            idx_to_char: Index to character mapping (for char mode)
            vocab_size: Vocabulary size
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.char_mode = char_mode
        self.char_to_idx = char_to_idx
        self.idx_to_char = idx_to_char
        self.vocab_size = vocab_size
        
        # Common English words and patterns for better generation
        self.common_words = [
            "the", "and", "a", "to", "of", "in", "is", "that", "it", "with",
            "for", "as", "on", "was", "at", "by", "an", "be", "this", "which"
        ]
        
        # Common sentence starters
        self.sentence_starters = [
            "The", "In", "When", "It", "There", "Although", "While", "After",
            "Before", "Since", "As", "If", "Though", "Because", "However"
        ]
        
        # Common punctuation patterns
        self.punctuation_patterns = {
            ".": [" ", "\n"],
            ",": [" "],
            "?": [" ", "\n"],
            "!": [" ", "\n"],
            ";": [" "],
            ":": [" "]
        }
        
    def sample_with_temperature(self, preds, temperature=1.0):
        """
        Sample with temperature and additional heuristics
        
        Args:
            preds: Model predictions
            temperature: Temperature for sampling (higher = more random)
            
        Returns:
            Sampled index
        """
        # Apply temperature
        preds = np.asarray(preds).astype('float64')
        
        # Handle very low temperatures (more deterministic)
        if temperature < 0.1:
            return np.argmax(preds)
        
        # Apply temperature scaling with clipping to avoid numerical issues
        preds = np.log(np.clip(preds, 1e-10, 1.0)) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        
        # For very high temperatures, add some bias toward more common tokens
        if temperature > 1.5:
            # Add a small bias to the top-k predictions
            top_indices = np.argsort(preds)[-5:]  # Get indices of top 5 predictions
            for idx in top_indices:
                preds[idx] *= 1.2  # Boost probability
            preds = preds / np.sum(preds)  # Renormalize
        
        # Sample from the distribution
        try:
            probas = np.random.multinomial(1, preds, 1)
            return np.argmax(probas)
        except ValueError:
            # Fallback if there's a sampling error
            print("Sampling error, using argmax instead")
            return np.argmax(preds)
    
    def generate_text_char_level(self, seed_text, length=100, temperature=0.5):
        """
        Generate text at character level with improved quality
        
        Args:
            seed_text: Seed text to start generation
            length: Length of text to generate
            temperature: Temperature for sampling
            
        Returns:
            Generated text
        """
        if not self.char_mode:
            raise ValueError("Model was not trained at character level")
        
        # Ensure seed text is at least max_sequence_length
        if len(seed_text) < self.max_sequence_length:
            seed_text = seed_text.rjust(self.max_sequence_length)
        
        # Take last max_sequence_length characters as seed
        generated_text = seed_text[-self.max_sequence_length:]
        
        # Track context for better generation
        last_chars = ""
        in_quote = False
        sentence_count = 0
        paragraph_count = 0
        capitalization_needed = False
        
        # Generate text
        for i in range(length):
            # Prepare input
            x = np.zeros((1, self.max_sequence_length, self.vocab_size))
            for t, char in enumerate(generated_text[-self.max_sequence_length:]):
                if char in self.char_to_idx:
                    x[0, t, self.char_to_idx[char]] = 1
            
            # Predict next character
            preds = self.model.predict(x, verbose=0)[0]
            
            # Apply context-aware adjustments
            last_chars = generated_text[-10:]
            
            # Handle quotes
            if '"' in last_chars:
                quote_count = last_chars.count('"')
                in_quote = quote_count % 2 == 1  # Odd number of quotes means we're inside quotes
            
            # Boost probability of space after punctuation
            if last_chars[-1] in ".!?":
                sentence_count += 1
                capitalization_needed = True
                if " " in self.char_to_idx:
                    space_idx = self.char_to_idx[" "]
                    preds[space_idx] *= 3.0  # Higher boost for space after end of sentence
                    preds = preds / np.sum(preds)
            
            # Boost probability of capitalization after period and space
            if capitalization_needed and last_chars[-1] == " ":
                # Find indices of uppercase letters
                for char, idx in self.char_to_idx.items():
                    if char.isupper():
                        preds[idx] *= 2.5  # Boost uppercase letters
                preds = preds / np.sum(preds)
                capitalization_needed = False
            
            # Boost comma probability for longer sentences
            if len(last_chars.split(".")[-1]) > 30 and last_chars[-1] not in ".!?,;:":
                if "," in self.char_to_idx:
                    comma_idx = self.char_to_idx[","]
                    preds[comma_idx] *= 1.5
                    preds = preds / np.sum(preds)
            
            # Sample with temperature
            next_index = self.sample_with_temperature(preds, temperature)
            next_char = self.idx_to_char[next_index]
            
            # Add to generated text
            generated_text += next_char
            
            # Add newline after multiple sentences for readability
            if sentence_count >= 3 and last_chars[-1] in ".!?" and next_char == " ":
                if np.random.random() < 0.4:  # 40% chance of newline after 3+ sentences
                    generated_text += "\n"
                    sentence_count = 0
                    paragraph_count += 1
            
            # Add double newline for new paragraph after several paragraphs
            if paragraph_count >= 2 and sentence_count >= 2 and last_chars[-1] in ".!?" and next_char == " ":
                if np.random.random() < 0.3:  # 30% chance of new paragraph
                    generated_text += "\n\n"
                    sentence_count = 0
                    paragraph_count = 0
        
        return generated_text
    
    def generate_text_word_level(self, seed_text, length=50, temperature=0.5):
        """
        Generate text at word level with improved quality
        
        Args:
            seed_text: Seed text to start generation
            length: Number of words to generate
            temperature: Temperature for sampling
            
        Returns:
            Generated text
        """
        if self.char_mode:
            raise ValueError("Model was not trained at word level")
        
        # Tokenize seed text
        seed_seq = self.tokenizer.texts_to_sequences([seed_text])[0]
        
        # Ensure seed sequence is at most max_sequence_length - 1
        if len(seed_seq) > self.max_sequence_length - 1:
            seed_seq = seed_seq[-(self.max_sequence_length - 1):]
        
        # Generate text
        generated_text = seed_text
        
        # Track context for better generation
        sentence_count = 0
        paragraph_count = 0
        last_punctuation = None
        words_since_comma = 0
        in_dialogue = False
        
        # Get common words from tokenizer
        common_words = []
        if hasattr(self.tokenizer, 'word_counts'):
            # Get top 100 most common words
            sorted_words = sorted(self.tokenizer.word_counts.items(), key=lambda x: x[1], reverse=True)[:100]
            common_words = [word for word, _ in sorted_words]
        
        for i in range(length):
            # Pad sequence
            padded_seq = pad_sequences([seed_seq], maxlen=self.max_sequence_length - 1)
            
            # Predict next word
            preds = self.model.predict(padded_seq, verbose=0)[0]
            
            # Apply context-aware adjustments
            words = generated_text.split()
            last_word = words[-1] if words else ""
            
            # Track dialogue
            if '"' in last_word:
                quote_count = sum(word.count('"') for word in words[-10:])
                in_dialogue = quote_count % 2 == 1  # Odd number of quotes means we're in dialogue
            
            # Boost common words after certain punctuation
            if last_word.endswith((".", "!", "?")):
                sentence_count += 1
                words_since_comma = 0
                last_punctuation = last_word[-1]
                
                # Boost probability of sentence starters
                for word in self.sentence_starters:
                    if word.lower() in self.tokenizer.word_index:
                        word_idx = self.tokenizer.word_index[word.lower()]
                        if word_idx < len(preds):
                            preds[word_idx] *= 2.0  # Increased boost
                
                # Normalize probabilities
                preds = preds / np.sum(preds)
            
            # Boost probability of comma after several words without punctuation
            words_since_comma += 1
            if words_since_comma > 10 and not in_dialogue:
                # Find words that often appear after commas
                for word in ["however", "therefore", "moreover", "nevertheless", "although", "meanwhile"]:
                    if word in self.tokenizer.word_index:
                        word_idx = self.tokenizer.word_index[word]
                        if word_idx < len(preds):
                            preds[word_idx] *= 1.5
                
                # Normalize probabilities
                preds = preds / np.sum(preds)
            
            # For dialogue, boost dialogue-appropriate words
            if in_dialogue:
                dialogue_words = ["said", "asked", "replied", "whispered", "shouted", "exclaimed"]
                if last_word.endswith('"'):
                    for word in dialogue_words:
                        if word in self.tokenizer.word_index:
                            word_idx = self.tokenizer.word_index[word]
                            if word_idx < len(preds):
                                preds[word_idx] *= 3.0
                    preds = preds / np.sum(preds)
            
            # Sample with temperature
            next_index = self.sample_with_temperature(preds, temperature)
            
            # Convert index to word
            next_word = ""
            for word, index in self.tokenizer.word_index.items():
                if index == next_index:
                    next_word = word
                    break
            
            # Add to generated text
            if next_word:
                # Add appropriate spacing
                if last_word and last_word[-1] in ".!?":
                    generated_text += " " + next_word.capitalize()
                elif last_word and last_word[-1] in ",;:":
                    generated_text += " " + next_word
                else:
                    generated_text += " " + next_word
                
                # Reset words_since_comma if we added a comma
                if next_word.endswith(","):
                    words_since_comma = 0
                
                # Add newline for readability after several sentences
                if sentence_count >= 3 and last_punctuation in ".!?" and not in_dialogue:
                    if np.random.random() < 0.4:  # 40% chance of newline after 3+ sentences
                        generated_text += "\n"
                        sentence_count = 0
                        paragraph_count += 1
                
                # Add paragraph break occasionally
                if paragraph_count >= 2 and sentence_count >= 2 and last_punctuation in ".!?" and not in_dialogue:
                    if np.random.random() < 0.3:  # 30% chance of paragraph break
                        generated_text += "\n\n"
                        sentence_count = 0
                        paragraph_count = 0
            
            # Update seed sequence
            seed_seq.append(next_index)
            if len(seed_seq) > self.max_sequence_length - 1:
                seed_seq = seed_seq[-(self.max_sequence_length - 1):]
        
        return generated_text
    
    def post_process_text(self, text):
        """
        Apply post-processing to fix common issues in generated text
        
        Args:
            text: Generated text to post-process
            
        Returns:
            Improved text
        """
        # Fix spacing around punctuation
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)
        
        # Fix spacing after punctuation
        text = re.sub(r'([.,;:!?])([^\s"\'])', r'\1 \2', text)
        
        # Fix quotes
        text = re.sub(r'\s+"', r' "', text)  # Space before opening quote
        text = re.sub(r'"\s+', r'" ', text)  # Space after closing quote
        
        # Ensure capitalization after periods
        def capitalize_after_period(match):
            return match.group(1) + match.group(2).upper() + match.group(3)
        
        text = re.sub(r'([.!?]\s+)([a-z])(\w*)', capitalize_after_period, text)
        
        # Fix repeated words
        text = re.sub(r'\b(\w+)\s+\1\b', r'\1', text)
        
        # Fix repeated punctuation (except for ellipsis)
        text = re.sub(r'([.!?])\1+', r'\1', text)
        text = re.sub(r'([,;:])\1+', r'\1', text)
        
        # Preserve ellipsis
        text = re.sub(r'\.\.\.+', '...', text)
        
        # Fix spacing around parentheses
        text = re.sub(r'\s+\)', r')', text)
        text = re.sub(r'\(\s+', r'(', text)
        
        # Fix multiple spaces
        text = re.sub(r'\s{2,}', ' ', text)
        
        # Fix spacing around newlines
        text = re.sub(r'\s+\n', '\n', text)
        text = re.sub(r'\n\s+', '\n', text)
        
        # Fix multiple newlines (more than 2)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text
    
    def generate_text(self, seed_text, length=100, temperature=0.5):
        """
        Generate text using the appropriate method with post-processing
        
        Args:
            seed_text: Seed text to start generation
            length: Length of text to generate
            temperature: Temperature for sampling
            
        Returns:
            Generated text
        """
        if self.char_mode:
            generated_text = self.generate_text_char_level(seed_text, length, temperature)
        else:
            generated_text = self.generate_text_word_level(seed_text, length, temperature)
        
        # Apply post-processing to improve the text quality
        return self.post_process_text(generated_text)
