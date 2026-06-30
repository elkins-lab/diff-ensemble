import nbformat as nbf

nb = nbf.v4.new_notebook()

nb.cells.append(
    nbf.v4.new_markdown_cell(
        """# 🧬 Quick Start: Differentiable IDP Ensemble Prediction

Use a Variational Autoencoder (VAE) to predict structural ensembles constrained by physical data.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elkins/diff-ensemble/blob/main/examples/quickstart_ensemble.ipynb)"""
    )
)

nb.cells.append(
    nbf.v4.new_code_cell(
        """# Setup for Colab or local run
import sys
if "google.colab" in sys.modules:
    !pip install -q diff-ensemble matplotlib
"""
    )
)

nb.cells.append(
    nbf.v4.new_markdown_cell(
        """## 1. Initialize the Ensemble VAE
We will initialize a generative model for a short 20-residue flexible peptide."""
    )
)

nb.cells.append(
    nbf.v4.new_code_cell(
        """import jax
import jax.numpy as jnp
from diff_ensemble.model import EnsembleVAE

# Define model parameters
seq_len = 20
latent_dim = 16
ensemble_size = 100

# Initialize the model
model = EnsembleVAE(seq_len=seq_len, latent_dim=latent_dim, ensemble_size=ensemble_size)

# Setup a random number generator key for JAX
rng = jax.random.PRNGKey(42)
rng, init_rng = jax.random.split(rng)

# Create dummy input features (1 sequence, seq_len residues, 20 amino acid features)
dummy_x = jnp.zeros((1, seq_len, 20))

# Initialize model parameters
params = model.init(init_rng, dummy_x, init_rng)
print(f"Model initialized! Ready to generate ensembles of {ensemble_size} structures.")
"""
    )
)

nb.cells.append(
    nbf.v4.new_markdown_cell(
        """## 2. Sample the Ensemble and Generate 3D Coordinates
We will run a forward pass through the un-trained VAE to sample a diverse ensemble of backbone conformations, and then use differentiable Neural Radiance Fields (NeRF) to build 3D Cartesian coordinates from the generated torsion angles."""
    )
)

nb.cells.append(
    nbf.v4.new_code_cell(
        """# Forward pass: Generate torsions (phi, psi) for the ensemble
rng, apply_rng = jax.random.split(rng)
torsions, mean, logvar = model.apply(params, dummy_x, apply_rng)

# Convert predicted torsions (ensemble_size, seq_len, 2) into 3D Cartesian coordinates
coords = model.generate_coordinates(torsions)

print(f"Generated Coordinates Shape: {coords.shape}")
print("(ensemble_size, seq_len * 3 atoms, 3 dimensions)")
"""
    )
)

nb.cells.append(
    nbf.v4.new_markdown_cell(
        """## 3. Visualize Structural Diversity
Let's overlay 5 randomly selected structures from the generated ensemble to see the flexibility of the predicted IDP."""
    )
)

nb.cells.append(
    nbf.v4.new_code_cell(
        """import matplotlib.pyplot as plt
import numpy as np

# Convert to NumPy for plotting
coords_np = np.array(coords)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the backbone trace of the first 5 structures
for i in range(5):
    # Extract only the C-alpha atoms (every 3rd atom starting at index 1: N, CA, C)
    ca_coords = coords_np[i, 1::3, :]
    ax.plot(ca_coords[:, 0], ca_coords[:, 1], ca_coords[:, 2], marker='o', label=f'Conf {i+1}')

ax.set_title("3D Overlay of Sampled IDP Conformations")
ax.set_xlabel("X (Å)")
ax.set_ylabel("Y (Å)")
ax.set_zlabel("Z (Å)")
ax.legend()
plt.show()
"""
    )
)

nb.cells.append(
    nbf.v4.new_markdown_cell(
        """## 4. Quantitative Analysis: Radius of Gyration ($R_g$)
To quantify the diversity of the ensemble, we can calculate the Radius of Gyration ($R_g$) for every generated conformation and plot the distribution. In a real scenario, this distribution could be optimized via backpropagation to match SAXS experimental data!"""
    )
)

nb.cells.append(
    nbf.v4.new_code_cell(
        """def calculate_rg(coords):
    \"\"\"Calculates the Radius of Gyration for a set of coordinates.\"\"\"
    # Center of mass (assuming equal mass for simplicity)
    com = np.mean(coords, axis=0)
    # Mean squared distance from COM
    sq_dist = np.sum((coords - com)**2, axis=1)
    return np.sqrt(np.mean(sq_dist))

# Calculate Rg for all 100 structures (using C-alpha atoms only)
ca_ensemble = coords_np[:, 1::3, :]
rg_values = [calculate_rg(ca_ensemble[i]) for i in range(ensemble_size)]

# Plot the distribution
plt.figure(figsize=(8, 5))
plt.hist(rg_values, bins=15, color='skyblue', edgecolor='black', alpha=0.7)
plt.axvline(np.mean(rg_values), color='red', linestyle='dashed', linewidth=2, label=f'Mean Rg: {np.mean(rg_values):.2f} Å')
plt.title(f"Distribution of Radius of Gyration ($R_g$) across the Ensemble")
plt.xlabel("Radius of Gyration (Å)")
plt.ylabel("Frequency")
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.show()
"""
    )
)

# Write the notebook
with open("examples/quickstart_ensemble.ipynb", "w") as f:
    nbf.write(nb, f)

print("Notebook generated successfully!")
