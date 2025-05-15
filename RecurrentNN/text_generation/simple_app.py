import streamlit as st
import numpy as np
import os
import re
import sys

# Monkey patch the TensorFlow plugin loading
def patch_tensorflow_loader():
    import importlib.machinery
    import importlib.util
    
    # First, we'll create a fake TensorFlow module
    class FakeTF:
        def __init__(self):
            self.keras = type('FakeKeras', (), {'preprocessing': type('FakePreprocessing', (), {'text': type('FakeText', (), {'Tokenizer': type('FakeTokenizer', (), {})}), 'sequence': type('FakeSequence', (), {'pad_sequences': lambda x: x})})()})
    
    # Now, create a finder that will return our fake module
    class FakeFinder:
        @staticmethod
        def find_spec(fullname, path, target=None):
            if fullname == 'tensorflow' or fullname.startswith('tensorflow.'):
                # Create a spec for our fake module
                loader = importlib.machinery.SourceFileLoader(fullname, __file__)
                spec = importlib.util.spec_from_loader(fullname, loader)
                spec.loader = FakeLoader()
                return spec
            return None
    
    # And a loader that will return our fake module
    class FakeLoader:
        def create_module(self, spec):
            return FakeTF()
        
        def exec_module(self, module):
            pass
    
    # Install our finder
    sys.meta_path.insert(0, FakeFinder)

# Apply the patch
patch_tensorflow_loader()

# Now we can safely import tensorflow
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Dummy text generator for demonstration
class SimpleTextGenerator:
    def __init__(self):
        self.char_mode = True
        self.vocab_size = 50
        self.max_sequence_length = 40
        
        # Sample vocabulary for demonstration
        self.chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?'\"-+=/\\:;()[]{} \n"
        self.char_to_idx = {char: i for i, char in enumerate(self.chars)}
        self.idx_to_char = {i: char for i, char in enumerate(self.chars)}
        
        # Sentence starters for word-level generation
        self.sentence_starters = ["The", "In", "When", "Although", "Because", "Since", "While"]
        
    def sample_with_temperature(self, preds, temperature=1.0):
        """Sample an index from a probability array with temperature"""
        if temperature <= 0:
            return np.argmax(preds)
        
        # Apply temperature
        preds = np.asarray(preds).astype('float64')
        preds = np.log(np.maximum(preds, 1e-10)) / temperature
        exp_preds = np.exp(preds - np.max(preds))
        preds = exp_preds / np.sum(exp_preds)
        
        # Sample from the distribution
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)
    
    def generate_text(self, seed_text, length=100, temperature=0.5):
        """Generate text using a simple Markov-like approach"""
        if len(seed_text) < self.max_sequence_length:
            seed_text = seed_text.rjust(self.max_sequence_length)
        
        generated_text = seed_text[-self.max_sequence_length:]
        
        # Simple character-based generation
        for i in range(length):
            # Create a simple probability distribution based on the last character
            last_char = generated_text[-1]
            preds = np.ones(len(self.chars)) * 0.01  # Base probability
            
            # Increase probability for characters that often follow the last character
            if last_char in "abcdefghijklmnopqrstuvwxyz":
                # After lowercase, likely lowercase or space
                for c in "abcdefghijklmnopqrstuvwxyz ":
                    if c in self.char_to_idx:
                        preds[self.char_to_idx[c]] = 0.1
            
            elif last_char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                # After uppercase, likely lowercase
                for c in "abcdefghijklmnopqrstuvwxyz":
                    if c in self.char_to_idx:
                        preds[self.char_to_idx[c]] = 0.1
            
            elif last_char in ".!?":
                # After punctuation, likely space then uppercase
                if " " in self.char_to_idx:
                    preds[self.char_to_idx[" "]] = 0.8
            
            elif last_char == " ":
                # After space, likely lowercase or uppercase
                for c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ":
                    if c in self.char_to_idx:
                        preds[self.char_to_idx[c]] = 0.1
                
                # Higher probability for uppercase after period + space
                if generated_text[-2] in ".!?":
                    for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                        if c in self.char_to_idx:
                            preds[self.char_to_idx[c]] = 0.3
            
            # Normalize probabilities
            preds = preds / np.sum(preds)
            
            # Sample with temperature
            next_index = self.sample_with_temperature(preds, temperature)
            next_char = self.idx_to_char[next_index]
            
            # Add to generated text
            generated_text += next_char
            
            # Add newline occasionally after periods for readability
            if len(generated_text) > 50 and next_char == " " and generated_text[-2] in ".!?":
                if np.random.random() < 0.2:
                    generated_text += "\n"
        
        return self.post_process_text(generated_text)
    
    def post_process_text(self, text):
        """Apply post-processing to fix common issues in generated text"""
        # Fix spacing around punctuation
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)
        
        # Fix spacing after punctuation
        text = re.sub(r'([.,;:!?])([^\s"\'])', r'\1 \2', text)
        
        # Fix quotes
        text = re.sub(r'\s+"', r' "', text)
        text = re.sub(r'"\s+', r'" ', text)
        
        # Ensure capitalization after periods
        def capitalize_after_period(match):
            return match.group(1) + match.group(2).upper() + match.group(3)
        
        text = re.sub(r'([.!?]\s+)([a-z])(\w*)', capitalize_after_period, text)
        
        # Fix repeated words
        text = re.sub(r'\b(\w+)\s+\1\b', r'\1', text)
        
        # Fix repeated punctuation
        text = re.sub(r'([.!?])\1+', r'\1', text)
        text = re.sub(r'([,;:])\1+', r'\1', text)
        
        # Fix multiple spaces
        text = re.sub(r'\s{2,}', ' ', text)
        
        return text

# Set up Streamlit app
st.set_page_config(
    page_title="Text Generation",
    page_icon="📝",
    layout="wide"
)

# Initialize session state
if 'generator' not in st.session_state:
    st.session_state.generator = SimpleTextGenerator()

# App title and description
st.title("📝 Text Generation Demo")
st.markdown("""
This is a simplified text generation demo that works around TensorFlow compatibility issues.
Enter some seed text and adjust the parameters to generate text.
""")

# Create sidebar for parameters
with st.sidebar:
    st.header("Generation Parameters")
    
    # Text generation parameters
    temperature = st.slider(
        "Temperature", 
        min_value=0.1, 
        max_value=1.5, 
        value=0.7, 
        step=0.1,
        help="Higher values make the text more random, lower values make it more deterministic"
    )
    
    length = st.slider(
        "Generation Length",
        min_value=50,
        max_value=500,
        value=200,
        step=50,
        help="Number of characters to generate"
    )

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input")
    seed_text = st.text_area(
        "Seed Text", 
        value="Once upon a time, there was a",
        height=150
    )
    
    generate_button = st.button("Generate Text", type="primary")

with col2:
    st.subheader("Generated Text")
    
    if generate_button or ('generated_text' in st.session_state and st.session_state.generated_text):
        with st.spinner("Generating text..."):
            generated_text = st.session_state.generator.generate_text(
                seed_text=seed_text,
                length=length,
                temperature=temperature
            )
            st.session_state.generated_text = generated_text
        
        st.text_area("Output", value=generated_text, height=300)
    else:
        st.info("Click 'Generate Text' to see the output")

# Footer
st.markdown("---")
st.markdown("This is a simplified version of the text generation app that works around TensorFlow compatibility issues.")
