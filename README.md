# Machine Learning Symmetry Discovery

This repository contains code for discovering symmetries in physical systems using machine learning techniques, with a focus on Hamiltonian systems like the Harmonic Oscillator and Kepler Problem.

## Overview

The project implements a Machine Learning Symmetry Discovery (MLSD) framework that learns conserved quantities and Lie algebra structures from data. It can identify symmetries in physical systems without prior knowledge of their mathematical form.

## Key Components

### MLSD.py
Core implementation containing:
- Neural network architectures for learning observables
- Poisson bracket computations
- Lie algebra structure discovery
- Loss functions for training

Key classes:
- `MLSD`: Main class for symmetry discovery
- `M_net`: Quadratic neural network for observables
- `Observable`: Wrapper for observable functions
- `HMCSampler`: Hamiltonian Monte Carlo sampler

### Data Preparation
Two notebooks for generating training data:
- `HO_data_preparation.ipynb`: Harmonic oscillator system
- `Kep_data_preparation.ipynb`: Kepler problem

Features:
- Numerical integration of equations of motion
- Energy conservation checks
- Phase space sampling
- Orbit visualization

### Main Examples
`main.ipynb` demonstrates:
- Training MLSD on the harmonic oscillator
- Visualizing learned structure constants
- Analyzing degeneracy of Killing forms

## Usage

1. Generate training data using the preparation notebooks
2. Initialize MLSD with appropriate parameters:
