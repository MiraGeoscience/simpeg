"""
PF: Gravity: Tiled Inversion Linear
===================================

Invert data in tiles.

"""
import numpy as np
import matplotlib.pyplot as plt

from SimPEG import (
    maps,
    data,
    data_misfit,
    regularization,
    optimization,
    inverse_problem,
    directives,
    inversion,
)
from SimPEG.utils.drivers.simulation_driver import SetupExample, GravitySimulationDriver
from SimPEG import utils

###############################################################################
# Setup
# -----
#
# Define the survey and model parameters
#
# Create a global survey and mesh and simulate some data
#
#
driver = SetupExample(GravitySimulationDriver).driver
model = np.zeros(driver.mesh.nC)
ind = utils.model_builder.getIndicesBlock(
    np.r_[-10, -10, -30],
    np.r_[10, 10, -10],
    driver.mesh.gridCC,
)[0]

# Assign magnetization values
model[ind] = 0.3

# Remove air cells
model = model[driver.active_cells]

###############################################################################
# Generate data and a misfit function
# Compute linear forward operator and compute some data
d = driver.simulation.fields(model)

# Add noise and uncertainties
# We add some random Gaussian noise (1nT)
synthetic_data = d + np.random.randn(len(d)) * 1e-3
wd = np.ones(len(synthetic_data)) * 1e-3  # Assign flat uncertainties
data_object = data.Data(
    driver.survey,
    dobs=synthetic_data,
    standard_deviation=wd,
)
misfit_function = data_misfit.L2DataMisfit(
    data=data_object, simulation=driver.simulation
)

###############################################################
# Create an inversion problem with regularization function
#
# Create a regularization
reg = regularization.Sparse(driver.mesh, active_cells=driver.active_cells)

# Add directives to the inversion
opt = optimization.ProjectedGNCG(
    maxIter=100, lower=-1.0, upper=1.0, maxIterLS=20, maxIterCG=10, tolCG=1e-3
)
invProb = inverse_problem.BaseInvProblem(misfit_function, reg, opt)
betaest = directives.BetaEstimate_ByEig(beta0_ratio=1e-1)

# Here is where the norms are applied
# Use a threshold parameter empirically based on the distribution of
# model parameters
update_IRLS = directives.Update_IRLS(
    f_min_change=1e-4,
    max_irls_iterations=0,
    coolEpsFact=1.5,
    beta_tol=1e-2,
)
saveDict = directives.SaveOutputEveryIteration(save_txt=False)
update_Jacobi = directives.UpdatePreconditioner()
sensitivity_weights = directives.UpdateSensitivityWeights(everyIter=False)
inv = inversion.BaseInversion(
    invProb,
    directiveList=[update_IRLS, sensitivity_weights, betaest, update_Jacobi, saveDict],
)

# Run the inversion
m0 = np.ones(driver.n_actives) * 1e-4  # Starting model
mrec = inv.run(m0)

###############################################################################
# Plot the true and recovered model with respective data maps.
ax = plt.subplot(2, 2, 1)
utils.plot_utils.plot2Ddata(driver.locations, synthetic_data, ax=ax, clim=[-1e-3, 1e-3])

ax = plt.subplot(2, 2, 2)
utils.plot_utils.plot2Ddata(
    driver.locations, driver.simulation.fields(mrec), ax=ax, clim=[-1e-3, 1e-3]
)

# Create active map to go from reduce set to full
inject_global = maps.InjectActiveCells(driver.mesh, driver.active_cells, np.nan)

ax = plt.subplot(2, 2, 3)
driver.mesh.plot_slice(inject_global * model, normal="Y", ax=ax, grid=True)
ax.set_title("True")
ax.set_aspect("equal")

ax = plt.subplot(2, 2, 4)
driver.mesh.plot_slice(inject_global * mrec, normal="Y", ax=ax, grid=True)
ax.set_title("Recovered")
ax.set_aspect("equal")
plt.show()
