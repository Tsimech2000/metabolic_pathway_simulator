import streamlit as st
import numpy as np
from scipy.integrate import odeint
import plotly.graph_objects as go
from deap import base, creator, tools, algorithms
import random

# Define a function to simulate a more complex metabolic pathway
def metabolic_pathway(k1, k2, k3, k_feedback, S0, t_max):
    # Define system of ODEs for the metabolic pathway with feedback inhibition
    def pathway_odes(y, t, k1, k2, k3, k_feedback):
        S, I, P, F = y  # S = Substrate, I = Intermediate, P = Product, F = Feedback Inhibitor
        dS_dt = -k1 * S * (1 - k_feedback * F)  # Feedback inhibition on enzyme 1
        dI_dt = k1 * S - k2 * I
        dP_dt = k2 * I - k3 * P
        dF_dt = k3 * P - 0.1 * F  # Feedback molecule accumulates but degrades over time
        return [dS_dt, dI_dt, dP_dt, dF_dt]
    
    # Initial conditions
    y0 = [S0, 0.0, 0.0, 0.0]
    t = np.linspace(0, t_max, 500)
    
    # Solve ODEs
    sol = odeint(pathway_odes, y0, t, args=(k1, k2, k3, k_feedback))
    return t, sol.T

def ai_optimization():
    st.header("AI-Driven Metabolic Pathway Optimization")
    st.write("Using a Genetic Algorithm (GA) to optimize enzyme reaction rates for maximum product yield.")
    
    # Define fitness function for optimization
    def fitness_function(individual):
        k1, k2, k3, k_feedback = individual
        _, results = metabolic_pathway(k1, k2, k3, k_feedback, S0=5.0, t_max=100)
        P_final = results[2, -1]  # Final product concentration
        return (P_final,)
    
    # Set up Genetic Algorithm with DEAP
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, 0.1, 5.0)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=4)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", fitness_function)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Run optimization
    pop = toolbox.population(n=20)
    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=10, verbose=False)
    best_ind = tools.selBest(pop, k=1)[0]
    
    # Run simulation with best parameters
    t, sol = metabolic_pathway(*best_ind, S0=5.0, t_max=100)
    S, I, P, F = sol
    
    # Display best parameters
    st.write(f"Optimized Parameters: k1={best_ind[0]:.2f}, k2={best_ind[1]:.2f}, k3={best_ind[2]:.2f}, k_feedback={best_ind[3]:.2f}")
    
    # Interactive Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=S, mode='lines', name='Substrate (S)'))
    fig.add_trace(go.Scatter(x=t, y=I, mode='lines', name='Intermediate (I)'))
    fig.add_trace(go.Scatter(x=t, y=P, mode='lines', name='Product (P)'))
    fig.add_trace(go.Scatter(x=t, y=F, mode='lines', name='Feedback Inhibitor (F)', line=dict(dash='dot')))
    fig.update_layout(title="Optimized Metabolic Pathway Simulation", xaxis_title="Time", yaxis_title="Concentration")
    
    st.plotly_chart(fig)

# Streamlit App
st.title("Metabolic Pathway Designer & AI Optimizer")

# Sidebar for navigation
simulation_choice = st.sidebar.selectbox(
    "Choose a Simulation:",
    ("Advanced Metabolic Pathway Simulation", "AI-Driven Optimization"))

# Load and run the selected simulation
dispatcher = {
    "Advanced Metabolic Pathway Simulation": lambda: run_metabolic_simulation(),
    "AI-Driven Optimization": ai_optimization,
}

dispatcher[simulation_choice]()
