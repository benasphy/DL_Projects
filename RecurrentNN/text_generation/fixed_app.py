import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import os

# Import the improved text generator
try:
    from improved_generator import ImprovedTextGenerator
    print("Successfully imported ImprovedTextGenerator")
except Exception as e:
    print(f"Error importing ImprovedTextGenerator: {e}")
    # Fallback to a simple text generator
    class MarkovTextGenerator:
        def __init__(self):
            # Load sample text for training the Markov model
            self.sample_texts = [
                "Once upon a time, there was a young prince who lived in a grand castle. The prince was known throughout the kingdom for his kindness and wisdom. Every day, he would walk through the village and talk to his people, listening to their concerns and offering help where he could. The villagers loved their prince and were grateful for his compassion.",
                "The sun was setting behind the mountains, casting long shadows across the valley. Birds were returning to their nests, singing their evening songs. A gentle breeze rustled the leaves of the ancient oak trees that stood like sentinels along the winding river. It was a peaceful scene, one that had remained unchanged for centuries.",
                "Sarah opened her eyes to the sound of rain pattering against the window. She loved rainy days; they gave her an excuse to stay inside with a good book and a cup of hot chocolate. She stretched lazily and reached for her favorite novel on the bedside table. Today would be a perfect day for reading and relaxation.",
                "The detective examined the crime scene carefully, noting every detail. Something wasn't right, but he couldn't quite put his finger on it. The room appeared undisturbed, yet the valuable painting was missing. There were no signs of forced entry, which suggested the thief had a key. This case was becoming more intriguing by the minute.",
                "The spaceship hurtled through the cosmos, its engines glowing with blue fire. Captain Rodriguez checked the navigation system and confirmed they were on course for Alpha Centauri. The journey would take five years, but the crew was prepared. They had trained for this mission their entire lives, and nothing would stop them from reaching humanity's first colony among the stars."
            ]
            
            # Build the Markov chain model
            self.build_markov_model()
            
            # Common sentence starters
            self.sentence_starters = ["The", "In", "When", "Although", "Because", "Since", "While", "After", "Before", "If", "She", "He", "They", "We", "I", "It", "There", "This", "That", "These", "Those"]
        
        def build_markov_model(self):
            """Build a Markov chain model from sample texts"""
            self.word_model = {}
            self.char_model = {}
            
            # Process each sample text
            for text in self.sample_texts:
                # Word-level model (2-gram)
                words = text.split()
                for i in range(len(words) - 2):
                    key = (words[i], words[i + 1])
                    if key not in self.word_model:
                        self.word_model[key] = []
                    self.word_model[key].append(words[i + 2])
                
                # Character-level model (3-gram)
                for i in range(len(text) - 3):
                    key = text[i:i+3]
                    if key not in self.char_model:
                        self.char_model[key] = []
                    self.char_model[key].append(text[i+3])
        
        def sample_with_temperature(self, options, temperature=1.0):
            """Sample from options with temperature"""
            if not options:
                return None
                
            if temperature <= 0.1:
                # Just pick the most common option
                return max(set(options), key=options.count)
            
            # Count occurrences of each option
            counts = {}
            for option in options:
                if option not in counts:
                    counts[option] = 0
                counts[option] += 1
            
            # Convert counts to probabilities and apply temperature
            options_list = list(counts.keys())
            probs = np.array([counts[opt] for opt in options_list], dtype=np.float64)
            probs = np.log(probs + 1e-10) / temperature
            probs = np.exp(probs - np.max(probs))
            probs = probs / np.sum(probs)
            
            # Sample from the distribution
            idx = np.random.choice(len(options_list), p=probs)
            return options_list[idx]
        
        def generate_text(self, seed_text, length=100, temperature=0.5):
            """Generate text using the Markov model"""
            if not seed_text:
                # Start with a random sentence starter if no seed text
                seed_text = np.random.choice(self.sentence_starters)
            
            generated_text = seed_text
            
            # Track context for better generation
            sentence_count = 0
            paragraph_count = 0
            capitalization_needed = False
            
            # Choose between character-level and word-level generation
            use_char_model = np.random.random() < 0.3  # 30% chance of character-level
            
            if use_char_model:
                # Character-level generation
                for _ in range(length):
                    # Get the last 3 characters as the key
                    if len(generated_text) < 3:
                        key = generated_text.ljust(3)
                    else:
                        key = generated_text[-3:]
                    
                    # Find possible next characters
                    if key in self.char_model:
                        options = self.char_model[key]
                        next_char = self.sample_with_temperature(options, temperature)
                    else:
                        # Fallback if key not found
                        if key[-1] in ".!?":
                            next_char = " "
                        elif key[-1] == " " and key[-2] in ".!?":
                            next_char = np.random.choice(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
                        elif key[-1] == " ":
                            next_char = np.random.choice(list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"))
                        else:
                            next_char = np.random.choice(list("abcdefghijklmnopqrstuvwxyz .,!?"))
                    
                    # Track sentence and paragraph structure
                    if next_char in ".!?":
                        sentence_count += 1
                        capitalization_needed = True
                    
                    # Add newline after multiple sentences for readability
                    if sentence_count >= 3 and next_char == " " and generated_text[-1] in ".!?":
                        if np.random.random() < 0.4:  # 40% chance of newline
                            generated_text += "\n"
                            sentence_count = 0
                            paragraph_count += 1
                            continue
                    
                    # Add paragraph break occasionally
                    if paragraph_count >= 2 and sentence_count >= 2 and next_char == " " and generated_text[-1] in ".!?":
                        if np.random.random() < 0.3:  # 30% chance of paragraph break
                            generated_text += "\n\n"
                            sentence_count = 0
                            paragraph_count = 0
                            continue
                    
                    # Add the next character
                    generated_text += next_char
            else:
                # Word-level generation
                words = generated_text.split()
                
                # Generate more words
                while len(generated_text.split()) < len(words) + length:
                    # Get the last two words as the key
                    if len(words) < 2:
                        # Not enough words yet, add a common starter
                        words.append(np.random.choice(self.sentence_starters))
                        continue
                    
                    key = (words[-2], words[-1])
                    
                    # Find possible next words
                    if key in self.word_model:
                        options = self.word_model[key]
                        next_word = self.sample_with_temperature(options, temperature)
                    else:
                        # Fallback if key not found
                        if words[-1][-1] in ".!?":
                            next_word = np.random.choice(self.sentence_starters)
                        else:
                            # Pick a random common word
                            common_words = ["the", "a", "an", "and", "but", "or", "in", "on", "at", "to", "with", "for", "of", "by", "as", "is", "was", "were", "are", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "should", "could", "may", "might", "must", "can"]
                            next_word = np.random.choice(common_words)
                    
                    # Add punctuation occasionally
                    if len(words) % 8 == 0 and np.random.random() < 0.3:
                        next_word += ","
                    
                    if len(words) % 15 == 0 and np.random.random() < 0.4:
                        next_word += "."
                        sentence_count += 1
                    
                    # Add the next word
                    words.append(next_word)
                    
                    # Check for sentence endings and paragraph breaks
                    if next_word[-1] in ".!?":
                        sentence_count += 1
                    
                    # Add paragraph break occasionally
                    if paragraph_count >= 2 and sentence_count >= 3 and next_word[-1] in ".!?":
                        if np.random.random() < 0.3:  # 30% chance of paragraph break
                            words.append("\n\n")
                            sentence_count = 0
                            paragraph_count = 0
                    
                    # Add newline after multiple sentences
                    elif sentence_count >= 3 and next_word[-1] in ".!?":
                        if np.random.random() < 0.4:  # 40% chance of newline
                            words.append("\n")
                            sentence_count = 0
                            paragraph_count += 1
                
                # Join words back into text
                generated_text = " ".join(words)
            
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
    st.session_state.generator = MarkovTextGenerator()

# App title and description
st.title("📝 Text Generation Demo")
st.markdown("""
### 🚧 Simple Project - Work in Progress 🚧

This is a simple text generation demo that uses a Markov chain model to generate text.
It's still under development and will be improved with more advanced techniques in the future.

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
        help="Higher values make the text more random, lower values make it more deterministic",
        key="temperature_slider"
    )
    
    length = st.slider(
        "Generation Length",
        min_value=50,
        max_value=500,
        value=200,
        step=50,
        help="Number of characters to generate",
        key="length_slider"
    )

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input")
    seed_text = st.text_area(
        "Seed Text", 
        value="Once upon a time, there was a",
        height=150,
        key="seed_text_area"
    )
    
    generate_button = st.button("Generate Text", type="primary", key="generate_button")

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
        
        st.text_area("Output", value=generated_text, height=300, key="output_text_area")
    else:
        st.info("Click 'Generate Text' to see the output")

# Footer
st.markdown("---")
st.markdown("""
### Project Status

This is a simplified version of the text generation app that works without TensorFlow dependencies. 
The current implementation uses a Markov chain model as a temporary solution while the TensorFlow compatibility issues are being resolved.

**Future Improvements:**
- Implement a proper deep learning model once TensorFlow issues are fixed
- Add more training data for better text generation
- Improve the user interface with more options and visualizations
- Add the ability to save and load generated texts

*Last updated: May 15, 2025*
""")
