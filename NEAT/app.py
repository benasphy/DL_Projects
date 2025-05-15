"""
Advanced Streamlit app for NEAT CartPole Neuroevolution
"""
import streamlit as st
from neat_cartpole import NEATCartpoleTrainer
import matplotlib.pyplot as plt

st.set_page_config(page_title="NEAT CartPole", layout="wide")
st.title("🧬 NEAT for CartPole-v1 (Neuroevolution)")

# Sidebar controls
generations = st.sidebar.slider("Generations", min_value=5, max_value=100, value=20, step=5)
run_training = st.sidebar.button('🚀 Train NEAT')
run_best = st.sidebar.button('🎮 Run Best Agent (no render)')

# Trainer session state
if 'neat_trainer' not in st.session_state:
    st.session_state.neat_trainer = NEATCartpoleTrainer()
    st.session_state.trained = False
    st.session_state.last_fitness = None

# Training section
if run_training:
    with st.spinner(f'Training NEAT for {generations} generations...'):
        st.session_state.neat_trainer = NEATCartpoleTrainer()
        winner = st.session_state.neat_trainer.train(generations=generations)
        st.session_state.trained = True
        st.success('Training complete!')

# Tabs for visualization
fitness_tab, agent_tab = st.tabs(["📈 Training Fitness", "🤖 Best Agent Demo"])

with fitness_tab:
    st.markdown("### Training Fitness Progress")
    if st.session_state.trained:
        fig = st.session_state.neat_trainer.plot_fitness()
        if fig:
            st.pyplot(fig)
        else:
            st.info("No training data available.")
    else:
        st.info("Train NEAT to see fitness curves.")

with agent_tab:
    st.markdown("### Run the Best Evolved Agent (no video)")
    if run_best and st.session_state.trained:
        fitness = st.session_state.neat_trainer.run_best(render=False)
        st.session_state.last_fitness = fitness
        st.success(f'Best agent achieved fitness: {fitness:.2f}')
    elif not st.session_state.trained:
        st.info("Train NEAT to run the best agent.")
    elif st.session_state.last_fitness is not None:
        st.success(f'Best agent achieved fitness: {st.session_state.last_fitness:.2f}')

# Footer
st.markdown('---')
st.markdown('Made with 🧬 NEAT-Python, Gym, and Streamlit')
