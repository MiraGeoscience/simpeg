"""
3D TEM Inversion with User-Defined Waveform on a Tree Mesh
==========================================================

Here we use the module *SimPEG.electromagnetics.time_domain* to predict the
TDEM response for a trapezoidal waveform. We consider an airborne survey
which uses a horizontal coplanar geometry. For this tutorial, we focus
on the following:

    - How to define the transmitters and receivers
    - How to define more complicated transmitter waveforms
    - How to define the time-stepping
    - How to define the survey
    - How to solve TDEM problems on an OcTree mesh
    - How to include topography
    - The units of the conductivity model and resulting data


Please note that we have used a coarse mesh and larger time-stepping to shorten
the time of the simulation. Proper discretization in space and time is required
to simulate the fields at each time channel with sufficient accuracy.


"""

#########################################################################
# Import Modules
# --------------
#

from discretize import TreeMesh
from discretize.utils import mkvc, refine_tree_xyz
from SimPEG import dask
from SimPEG.utils import plot2Ddata, surface2ind_topo, model_builder
from SimPEG import data, data_misfit, directives, maps, optimization, regularization, inverse_problem, inversion
import SimPEG.electromagnetics.time_domain as tdem
# from geoh5py.workspace import Workspace
# from geoh5py.objects import Curve
from geoapps.driver_base.utils import treemesh_2_octree
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from time import time
try:
    from pymatsolver import Pardiso as Solver
except ImportError:
    from SimPEG import SolverLU as Solver

save_file = False

# sphinx_gallery_thumbnail_number = 3

# ws = Workspace(f"./Test_{time():.0f}.geoh5")

###############################################################
# Defining Topography
# -------------------
#
# Here we define surface topography as an (N, 3) numpy array. Topography could
# also be loaded from a file. Here we define flat topography, however more
# complex topographies can be considered.
#

xx, yy = np.meshgrid(np.linspace(-6000, 6000, 101), np.linspace(-6000, 6000, 101))
zz = np.zeros(np.shape(xx))
topo_xyz = np.c_[mkvc(xx), mkvc(yy), mkvc(zz)]


###############################################################
# Defining the Waveform
# ---------------------
#
# Under *SimPEG.electromagnetic.time_domain.sources*
# there are a multitude of waveforms that can be defined (VTEM, Ramp-off etc...).
# Here, we consider a trapezoidal waveform, which consists of a
# linear ramp-on followed by a linear ramp-off. For each waveform, it
# is important you are cognizant of the off time!!!
#


# Define a discrete set of times for which your transmitter is 'on'. Here
# the waveform is on from -0.002 s to 0 s.
waveform_times = np.linspace(-0.002, 0, 21)

# For each waveform type, you must define the necessary set of kwargs.
# For the trapezoidal waveform we define the ramp on interval, the
# ramp-off interval and the off-time.
waveform = tdem.sources.TrapezoidWaveform(
    ramp_on=np.r_[-0.002, -0.0001], ramp_off=np.r_[-0.0001, 0.0], offTime=0.0
)

# Uncomment to try a quarter sine wave ramp on, followed by a linear ramp-off.
# waveform = tdem.sources.QuarterSineRampOnWaveform(
#     ramp_on=np.r_[-0.002, -0.001],  ramp_off=np.r_[-0.001, 0.], offTime=0.
# )

# Uncomment to try a custom waveform (just a linear ramp-off). This requires
# defining a function for your waveform.
# def wave_function(t):
#     return - t/(np.max(waveform_times) - np.min(waveform_times))
#
# waveform = tdem.sources.RawWaveform(waveFct=wave_function, offTime=0.)

# Evaluate the waveform for each on time.
waveform_value = [waveform.eval(t) for t in waveform_times]

# Plot the waveform
# fig = plt.figure(figsize=(10, 4))
# ax1 = fig.add_subplot(111)
# ax1.plot(waveform_times, waveform_value, lw=2)
# ax1.set_xlabel("Times [s]")
# ax1.set_ylabel("Waveform value")
# ax1.set_title("Waveform")


#####################################################################
# Create Airborne Survey
# ----------------------
#
# Here we define the survey used in our simulation. For time domain
# simulations, we must define the geometry of the source and its waveform. For
# the receivers, we define their geometry, the type of field they measure and
# the time channels at which they measure the field. For this example,
# the survey consists of a uniform grid of airborne measurements.
#

# Observation times for response (time channels)
n_times = 8
time_channels = np.logspace(-4, -2.5, n_times)
time_steps = [(5e-4, 4), (5e-5, 10), (5e-4, 10)]

# Defining transmitter locations
n_tx = 5
xtx, ytx, ztx = np.meshgrid(
    np.linspace(-200, 200, n_tx), np.linspace(-200, 200, n_tx), [50]
)
source_locations = np.c_[mkvc(xtx), mkvc(ytx), mkvc(ztx)]
ntx = np.size(xtx)

# Define receiver locations
xrx, yrx, zrx = np.meshgrid(
    np.linspace(-200, 200, n_tx), np.linspace(-190, 190, n_tx), [30]
)
receiver_locations = np.c_[mkvc(xrx), mkvc(yrx), mkvc(zrx)]

source_list = []  # Create empty list to store sources

# Each unique location defines a new transmitter
value_sortings = []
rx_counter = 0
for ii in range(ntx):

    # Here we define receivers that measure the h-field in A/m
    dbzdt_receiver = tdem.receivers.PointMagneticFluxTimeDerivative(
        receiver_locations[ii, :], time_channels, "z"
    )
    receivers_list = [
        dbzdt_receiver
    ]  # Make a list containing all receivers even if just one

    # Must define the transmitter properties and associated receivers
    source_list.append(
        tdem.sources.MagDipole(
            receivers_list,
            location=source_locations[ii],
            waveform=waveform,
            moment=1.0,
            orientation="z",
        )
        # tdem.sources.LineCurrent(
        #     receivers_list,
        #     location=np.r_[
        #         np.c_[-100, -100, 30],
        #         np.c_[-100, 100, 30],
        #         np.c_[100, 100, 30],
        #         np.c_[100, -100, 30],
        #         np.c_[-100, -100, 30],
        #     ],
        #     waveform=waveform,
        # )
    )
    nrx = dbzdt_receiver.locations.shape[0]
    for ti in range(n_times):
        value_sortings += [[ti, 0, ind] for ind in range(rx_counter, rx_counter + nrx)]

    rx_counter += nrx

value_sortings = np.vstack(value_sortings)
survey = tdem.Survey(source_list)
# tem_survey = Curve.create(ws, vertices=receiver_locations, name="TEM Survey")
###############################################################
# Create OcTree Mesh
# ------------------
#
# Here we define the OcTree mesh that is used for this example.
# We chose to design a coarser mesh to decrease the run time.
# When designing a mesh to solve practical time domain problems:
#
#     - Your smallest cell size should be 10%-20% the size of your smallest diffusion distance
#     - The thickness of your padding needs to be 2-3 times biggest than your largest diffusion distance
#     - The diffusion distance is ~1260*np.sqrt(rho*t)
#
#

dh = 25.0  # base cell width
dom_width = 4000.0  # domain width
nbc = 2 ** int(np.round(np.log(dom_width / dh) / np.log(2.0)))  # num. base cells

# Define the base mesh
h = [(dh, nbc)]
mesh = TreeMesh([h, h, h], x0="CCC")

# Mesh refinement based on topography
mesh = refine_tree_xyz(
    mesh, topo_xyz, octree_levels=[0, 0, 0, 1], method="surface", finalize=False
)

# Mesh refinement near transmitters and receivers
mesh = refine_tree_xyz(
    mesh, receiver_locations, octree_levels=[4, 4], method="radial", finalize=False
)

# Refine core mesh region
xp, yp, zp = np.meshgrid([-250.0, 250.0], [-250.0, 250.0], [-250.0, 0.0])
xyz = np.c_[mkvc(xp), mkvc(yp), mkvc(zp)]
mesh = refine_tree_xyz(mesh, xyz, octree_levels=[0, 2, 4], method="box", finalize=False)
mesh.finalize()

###############################################################
# Create Conductivity Model and Mapping for OcTree Mesh
# -----------------------------------------------------
#
# Here, we define the electrical properties of the Earth as a conductivity
# model. The model consists of a conductive block within a more
# resistive background.
#

# Conductivity in S/m
air_conductivity = np.log(1e-8)
background_conductivity = np.log(1e-3)
block_conductivity = np.log(1e+0)

# Active cells are cells below the surface.
ind_active = surface2ind_topo(mesh, topo_xyz)
nC = int(ind_active.sum())
model_map = maps.ExpMap() * maps.InjectActiveCells(mesh, ind_active, air_conductivity)

# Define the model
model = background_conductivity * np.ones(mesh.n_cells)
# face = np.r_[
#     np.c_[-50, -100, -50],
#     np.c_[-50, 100, -50],
#     np.c_[50, 100, -50],
#     np.c_[50, -100, -50]
# ]
#
# ind_block = model_builder.PolygonInd(
#     mesh,
#     np.r_[
#         face + np.c_[100, 0, 0],
#         face + np.c_[-150, 0, -225]
#     ]
# )
ind_block = model_builder.getIndicesBlock(np.r_[-50, -50, -150], np.r_[50, 50, -50], mesh.cell_centers)
model[ind_block] = block_conductivity
model = model[ind_active]

# Plot log-conductivity model
plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan) * maps.ExpMap()


def plot_model(model):
    plt.figure(figsize=(5, 7))

    ax1 = plt.subplot(2, 1, 2)
    im = mesh.plotSlice(
        model,
        ax=ax1,
        normal="Y",
        grid=True,
        clim=(-3, 0),
        pcolor_opts={"cmap": mpl.cm.viridis},
    )
    ax1.set_title("")
    ax1.set_xlabel("x (m)")
    plt.colorbar(im[0])
    ax1.set_ylabel("z (m)")
    ax1.set_xlim([-250, 250])
    ax1.set_ylim([-500, 0])
    ax1.set_aspect('equal')

    ax2 = plt.subplot(2, 1, 1)
    im = mesh.plotSlice(
        model,
        normal="Z",
        ax=ax2,
        grid=True,
        ind=60,
        clim=(-3, 0),
        pcolor_opts={"cmap": mpl.cm.viridis},
    )
    ax2.set_title("Recovered model")
    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("y (m)")
    plt.colorbar(im[0])
    ax2.set_xlim([-250, 250])
    ax2.set_ylim([-250, 250])
    ax2.set_aspect('equal')

    ax2.scatter(receiver_locations[:, 0], receiver_locations[:, 1], 10, color='r')
    plt.show()


plot_model(np.log10(plotting_map * model))

#######################################################################
# Simulation: Time-Domain Response
# --------------------------------
#
# Here we define the formulation for solving Maxwell's equations. Since we are
# measuring the time-derivative of the magnetic flux density and working with
# a resistivity model, the EB formulation is the most natural. We must also
# remember to define the mapping for the conductivity model.
# We defined a waveform 'on-time' is from -0.002 s to 0 s. As a result, we need
# to set the start time for the simulation to be at -0.002 s.
#

simulation = tdem.simulation.Simulation3DMagneticFluxDensity(
    mesh, survey=survey, sigmaMap=model_map, solver=Solver, t0=-0.002
)

# Set the time-stepping for the simulation
simulation.time_steps = time_steps

########################################################################
# Predict Data and Plot
# ---------------------
#

def reshape(values):
    data = np.zeros((len(time_channels), 1, receiver_locations.shape[0]))
    data[value_sortings[:, 0], value_sortings[:, 1], value_sortings[:, 2]] = values
    return data


# Predict data for a given model
dpred = simulation.dpred(model)
# floors = (
#     np.ones_like(reshape(np.abs(dpred))) *
#     np.max(reshape(np.abs(dpred)), axis=2).flatten()[:, None, None] / 4.
# ) + 1e-15

floors = np.abs(simulation.dpred(np.ones_like(model) * np.log(5e-3)))
noise = np.random.randn(dpred.shape[0]) * ( #1e-15)
            np.abs(dpred) * 0.02
)
data_object = data.Data(
    survey,
    dobs=dpred + noise,
    noise_floor=floors.flatten(order='F'),
)
dmis = data_misfit.L2DataMisfit(simulation=simulation, data=data_object, model_map=maps.IdentityMap(nP=nC))
reg = regularization.Sparse(
    mesh, alpha_s=0.,  # Set alpha_s = 0 to remove the reference model
    indActive=ind_active,
    mapping=maps.IdentityMap(nP=nC),
    gradientType="total",
    mref=np.log(1e-3) * np.ones(nC)
)
opt = optimization.ProjectedGNCG(
    maxIter=15,
    lower=-np.inf,
    upper=np.inf,
    maxIterLS=10,
    maxIterCG=40,
    tolCG=1e-4
)
inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)
inv = inversion.BaseInversion(
    inv_prob, directiveList=[
        # directives.UpdateSensitivityWeights(
        #     method="percent_amplitude",
        #     threshold=30.
        # ),
        # directives.SaveIterationsGeoH5(octree, transforms=[plotting_map], sorting=mesh._ubc_order),
        # directives.SaveIterationsGeoH5(
        #     tem_survey,
        #     attribute_type="predicted",
        #     channels=[f"{val:.2e}" for val in time_channels],
        #     association="VERTEX"
        # ),
        directives.Update_IRLS(
            max_irls_iterations=0,
            coolingRate=3,
            chifact_start=0.1,
            chifact_target=0.1,
        ),
        directives.UpdatePreconditioner(),
        directives.BetaEstimate_ByEig(beta0_ratio=1e+1, method="old")
    ]
)

# Run the inversion
m0 = np.log(2e-3) * np.ones(nC)
mrec = inv.run(m0)

# Plot recovered model
recovered_conductivity_model_log10 = np.log10(plotting_map * mrec)

plot_model(recovered_conductivity_model_log10)

#######################################################
# Optional: Export Data
# ---------------------
#
# Write the true model, data and topography
#

plt.figure()
axs = plt.subplot()
plt.plot(reshape(dpred).squeeze().T, "k")
plt.plot(reshape(data_object.dobs).squeeze().T, "b")
plt.plot(reshape(np.r_[inv_prob.dpred]).squeeze().T, "r")
plt.plot(reshape(data_object.standard_deviation).squeeze().T, "k--")
plt.plot(reshape(-data_object.standard_deviation).squeeze().T, "k--")
axs.set_yscale("symlog", linthresh=1e-13)
axs.set_title("Obs - Predicted")
axs.set_xlabel("Station ID #")
axs.set_ylabel("dBdt (T/s)")
axs.set_ylim([-1e-10, 0])

plt.show()
# if hasattr(inv_prob, "l2model"):
#     axs = plt.subplot(2,1,1)
#     plt.plot(reshape(dpred).squeeze().T, "k")
#     plt.plot(reshape(data_object.dobs).squeeze().T, "b")
#     plt.plot(reshape(simulation.dpred(inv_prob.l2model)).squeeze().T, "r")
#     plt.plot(reshape(data_object.standard_deviation).squeeze().T, "k--")
#     plt.plot(reshape(-data_object.standard_deviation).squeeze().T, "k--")
#     axs.set_yscale("symlog", linthresh=1e-14)
#     axs.set_title("L2 Predicted")
#     axs.set_ylim([-1e-10, 0])
#     plt.show()

if save_file:

    dir_path = os.path.dirname(tdem.__file__).split(os.path.sep)[:-3]
    dir_path.extend(["tutorials", "assets", "tdem"])
    dir_path = os.path.sep.join(dir_path) + os.path.sep

    fname = dir_path + "tdem_topo.txt"
    np.savetxt(fname, np.c_[topo_xyz], fmt="%.4e")

    # Write data with 2% noise added
    fname = dir_path + "tdem_data.obs"
    dpred = dpred + 0.02 * np.abs(dpred) * np.random.rand(len(dpred))
    t_vec = np.kron(np.ones(ntx), time_channels)
    receiver_locations = np.kron(receiver_locations, np.ones((len(time_channels), 1)))

    np.savetxt(fname, np.c_[receiver_locations, t_vec, dpred], fmt="%.4e")

    # Plot true model
    output_model = plotting_map * model
    output_model[np.isnan(output_model)] = 1e-8

    fname = dir_path + "true_model.txt"
    np.savetxt(fname, output_model, fmt="%.4e")
