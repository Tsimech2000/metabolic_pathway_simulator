import streamlit as st
import numpy as np
from scipy.integrate import odeint
import plotly.graph_objects as go
from scipy.optimize import differential_evolution
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

# AI Optimization using Scipy's Differential Evolution
def ai_optimization():
    st.header("AI-Driven Metabolic Pathway Optimization")
    st.write("Using Differential Evolution (Scipy) to optimize enzyme reaction rates for maximum product yield.")
    
    # Define fitness function for optimization
    def fitness_function(params):
        k1, k2, k3, k_feedback = params
        _, results = metabolic_pathway(k1, k2, k3, k_feedback, S0=5.0, t_max=100)
        P_final = results[2, -1]  # Final product concentration
        return -P_final  # Negative sign since we want to maximize
    
    # Run Differential Evolution Optimization
    bounds = [(0.1, 5.0), (0.1, 5.0), (0.1, 5.0), (0.0, 2.0)]
    result = differential_evolution(fitness_function, bounds, strategy='best1bin', maxiter=20, popsize=10)
    best_params = result.x
    
    # Run simulation with best parameters
    t, sol = metabolic_pathway(*best_params, S0=5.0, t_max=100)
    S, I, P, F = sol
    
    # Display best parameters
    st.write(f"Optimized Parameters: k1={best_params[0]:.2f}, k2={best_params[1]:.2f}, k3={best_params[2]:.2f}, k_feedback={best_params[3]:.2f}")
    
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
    "Advanced Metabolic Pathway Simulation": ai_optimization,
    "AI-Driven Optimization": ai_optimization,
}

dispatcher[simulation_choice]()

