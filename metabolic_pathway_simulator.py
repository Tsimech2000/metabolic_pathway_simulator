import streamlit as st
import numpy as np
from scipy.integrate import odeint
import plotly.graph_objects as go

# Define a function to simulate a more complex metabolic pathway
def metabolic_pathway():
    st.header("Advanced Metabolic Pathway Simulator")
    st.write("Simulate a complex metabolic pathway with multiple enzymatic reactions and feedback regulation.")
    
    # Sidebar inputs for reaction rates and initial concentrations
    k1 = st.sidebar.slider("Enzyme 1 Reaction Rate (k1)", 0.1, 5.0, 1.0)
    k2 = st.sidebar.slider("Enzyme 2 Reaction Rate (k2)", 0.1, 5.0, 1.0)
    k3 = st.sidebar.slider("Enzyme 3 Reaction Rate (k3)", 0.1, 5.0, 1.0)
    k_feedback = st.sidebar.slider("Feedback Inhibition Rate (k_feedback)", 0.0, 2.0, 0.5)
    S0 = st.sidebar.slider("Initial Substrate Concentration (S0)", 0.1, 10.0, 5.0)
    t_max = st.sidebar.slider("Simulation Time", 10, 200, 100)
    
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
    S, I, P, F = sol.T
    
    # Interactive Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=S, mode='lines', name='Substrate (S)'))
    fig.add_trace(go.Scatter(x=t, y=I, mode='lines', name='Intermediate (I)'))
    fig.add_trace(go.Scatter(x=t, y=P, mode='lines', name='Product (P)'))
    fig.add_trace(go.Scatter(x=t, y=F, mode='lines', name='Feedback Inhibitor (F)', line=dict(dash='dot')))
    fig.update_layout(title="Advanced Metabolic Pathway Simulation", xaxis_title="Time", yaxis_title="Concentration")
    
    st.plotly_chart(fig)

# Streamlit App
st.title("Metabolic Pathway Designer & Simulator")

# Sidebar for navigation
simulation_choice = st.sidebar.selectbox(
    "Choose a Simulation:",
    ("Advanced Metabolic Pathway Simulation",)
)

# Load and run the selected simulation
dispatcher = {
    "Advanced Metabolic Pathway Simulation": metabolic_pathway,
}

dispatcher[simulation_choice]()
