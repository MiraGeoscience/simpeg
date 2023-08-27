from abc import ABC, abstractmethod
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import cKDTree

from SimPEG.potential_fields import gravity
from SimPEG import maps
from SimPEG.survey import BaseSurvey
from discretize import TreeMesh
from discretize.utils import mesh_builder_xyz, refine_tree_xyz, active_from_xyz

from SimPEG.simulation import BaseSimulation
from SimPEG import utils


class OctreeDriver:
    """
    An octree mesh creation driver.

    Choices on the cell size, extent, depth_core and refinements are made
    based on the receiver locations and topography if not provided.
    """

    _depth_core: tuple[float]
    _locations: np.ndarray
    _padding_distance: tuple[float]
    _refinements: list[dict]

    def __init__(
        self,
        locations: np.ndarray | list,
        cell_size: tuple[float, float, float] | None = None,
        depth_core: float | None = None,
        padding_distance: tuple[float] = (0.0, 0.0, 0.0),
        refinements: list[dict] | None = None,
    ):
        self._mesh: TreeMesh | None = None
        self._cell_size: tuple[float, float, float] | None = None

        self.locations = locations
        self.cell_size = cell_size
        self.depth_core = depth_core
        self.padding_distance = padding_distance
        self.refinements = refinements

    @property
    def cell_size(self) -> tuple[float, float, float]:
        """
        The cell size of the mesh.

        Defaults to the median nearest neighbour distance of the locations.
        """
        if self._cell_size is None:
            tree = cKDTree(self.locations)
            dists, _ = tree.query(self.locations, k=2)
            self._cell_size = (np.median(dists[1]),) * 3

        return self._cell_size

    @cell_size.setter
    def cell_size(self, values):
        if not isinstance(values, (tuple, type(None))):
            raise TypeError(
                f"Attribute 'cell_size' must be of type tuple, not {type(values)}."
            )

        if not isinstance(values, type(None)):
            if len(values) != 3:
                raise ValueError(
                    f"Attribute 'cell_size' must be of length 3, not {len(values)}."
                )

        self._cell_size = values

    @property
    def depth_core(self):
        """
        The depth of the core region of the mesh.

        Uses the half width of the locations if not provided.
        """
        if self._depth_core is None:
            extent = self.locations.max(axis=0) - self.locations.min(axis=0)
            self._depth_core = np.max(extent) / 2.0

        return self._depth_core

    @depth_core.setter
    def depth_core(self, value):
        if not isinstance(value, (float, type(None))):
            raise TypeError(
                f"Attribute 'depth_core' must be of type float, not {type(value)}."
            )

        self._depth_core = value

    @property
    def locations(self) -> np.ndarray:
        """
        The receiver locations for the simulation.
        """
        return self._locations

    @locations.setter
    def locations(self, locations: np.ndarray | None):
        if not isinstance(locations, (np.ndarray, list, type(None))):
            raise TypeError(
                f"Attribute 'locations' must be of type np.ndarray, not {type(locations)}."
            )

        if isinstance(locations, list):
            locations = np.vstack(locations)

        if not isinstance(locations, type(None)):
            if locations.shape[1] != 3:
                raise ValueError(
                    f"Attribute 'locations' must be of shape (n, 3), not {locations.shape}."
                )

        self._locations = locations

    @property
    def mesh(self) -> TreeMesh:
        """
        An octree mesh object for the simulation.
        """
        if self._mesh is None:
            self._mesh = mesh_builder_xyz(
                self.locations,
                self.cell_size,
                depth_core=self.depth_core,
                mesh_type="tree",
            )

            for refinement in self.refinements:
                self._mesh = refine_tree_xyz(
                    self._mesh,
                    refinement["locations"],
                    method=refinement["type"],
                    octree_levels=refinement["levels"],
                    finalize=False,
                )

            self._mesh.finalize()

        return self._mesh

    @property
    def refinements(self):
        """
        The refinements to apply to the mesh.

        Defaults to 4 levels of 'surface' refinement on the locations.
        """
        if self._refinements is None:
            self._refinements = [
                {
                    "locations": self.locations,
                    "type": "surface",
                    "levels": (4, 4, 4),
                }
            ]
        return self._refinements

    @refinements.setter
    def refinements(self, refinements: dict | None):
        if not isinstance(refinements, (list, type(None))):
            raise TypeError(
                f"Attribute 'refinements' must be of type dict, not {type(refinements)}."
            )

        for value in refinements:
            if not isinstance(value, dict):
                raise TypeError(
                    f"Attribute 'refinements' must be of type dict, not {type(value)}."
                )
            if list(value) != ["locations", "type", "levels"]:
                raise KeyError(
                    "Attribute 'refinements' must contain 'locations', 'type' and 'levels'."
                )
            if not isinstance(value.get("locations", None), np.ndarray):
                raise TypeError("Attribute 'locations' must be of type np.ndarray.")

        self._refinements = refinements


class BaseOctreeSimulationDriver(ABC):
    _mesh_driver: OctreeDriver
    _receiver_locations: np.ndarray
    _simulation_class: BaseSimulation
    _survey_class: BaseSurvey
    _topography: np.ndarray

    """
    Base driver class for generating a simulation, survey and mesh for a given
    array or receivers and topography.


    Parameters
    ----------
    cell_size: tuple[float]
        The cell size of the mesh.
    receiver_locations: np.ndarray
        The receiver locations for the simulation.
    refinements: tuple[int]
        The number of refinements to apply to the mesh.
    topography: np.ndarray
        The topography locations defaulted to a Gaussian surface with twice


    Attributes
    ----------
    active_cells: np.ndarray
        Array of bool for the active cells.
    mesh: TreeMesh
        An octree mesh object for the simulation.
    simulation: BaseSimulation
        The simulation object.
    survey: BaseSurvey
        Generate the survey object for a given class type.
    """

    def __init__(
        self,
        receiver_locations: np.ndarray,
        topography: np.ndarray,
        mesh_driver: OctreeDriver | None = None,
    ):
        self._active_cells = None
        self._mesh: TreeMesh | None = None
        self._simulation: BaseSimulation | None = None
        self._survey: BaseSurvey | None = None

        self.mesh_driver = mesh_driver
        self.receiver_locations = receiver_locations
        self.topography: np.ndarray = topography

    @property
    def active_cells(self) -> np.ndarray:
        """
        Array of bool for the active cells.
        """
        if self._active_cells is None:
            self._active_cells = active_from_xyz(self.mesh, self.topography)

        return self._active_cells

    @property
    def mesh(self) -> TreeMesh:
        """
        An octree mesh object for the simulation.
        """
        if self._mesh is None:
            self._mesh = self.mesh_driver.mesh

        return self._mesh

    @property
    def mesh_driver(self):
        """
        The mesh driver.
        """
        if getattr(self, "_mesh_driver", None) is None:
            self._mesh_driver = OctreeDriver(
                self.locations,
                refinements=[
                    {
                        "locations": self.receiver_locations,
                        "type": "surface",
                        "levels": (4, 4, 4),
                    },
                    {
                        "locations": self.topography,
                        "type": "surface",
                        "levels": (0, 0, 4),
                    },
                ],
            )
        return self._mesh_driver

    @mesh_driver.setter
    def mesh_driver(self, driver: OctreeDriver):
        if not isinstance(driver, (OctreeDriver, type(None))):
            raise TypeError(
                f"Attribute 'mesh_driver' must be of type OctreeDriver, not {type(driver)}."
            )

        self._mesh_driver = driver

    @property
    def locations(self):
        """
        Pointer to the receiver and/or transmitter locations for the simulation.
        """
        if self.receiver_locations is None:
            raise AttributeError(
                "Attribute 'receiver_locations' must be defined before calling 'locations'."
            )

        return self.receiver_locations

    @property
    def n_actives(self) -> int:
        """
        The number of active cells.
        """
        return int(self.active_cells.sum())

    @property
    @abstractmethod
    def simulation(self):
        """
        The simulation object.
        """

    @property
    @abstractmethod
    def survey(self):
        """
        Generate the survey object for a given class type.
        """

    @property
    def receiver_locations(self):
        """
        The receiver locations for the simulation.
        """
        return self._receiver_locations

    @receiver_locations.setter
    def receiver_locations(self, locations: np.ndarray | None):
        if not isinstance(locations, (np.ndarray, type(None))):
            raise TypeError(
                f"Attribute 'receiver_locations' must be of type np.ndarray, not {type(locations)}."
            )

        self._receiver_locations = locations

    @property
    def topography(self):
        """
        The receiver locations for the simulation.
        """
        return self._topography

    @topography.setter
    def topography(self, locations: np.ndarray | None):
        if not isinstance(locations, (np.ndarray, type(None))):
            raise TypeError(
                f"Attribute 'topography' must be of type np.ndarray, not {type(locations)}."
            )

        self._topography = locations


class GravitySimulationDriver(BaseOctreeSimulationDriver):
    _simulation_class = gravity.simulation.Simulation3DIntegral
    _survey_class = gravity.survey.Survey

    """
    The gravity example driver.

    **kwargs
        See BaseExampleDriver.
    """

    def __init__(
        self, receiver_locations: np.ndarray, topography: np.ndarray, **kwargs
    ):
        super().__init__(receiver_locations, topography, **kwargs)

    @property
    def survey(self):
        """
        The gravity survey object.
        """
        if self._survey is None:
            receivers = gravity.receivers.Point(self.receiver_locations)
            sources = gravity.sources.SourceField([receivers])
            self._survey = self._survey_class(sources)

        return self._survey

    @property
    def simulation(self):
        """
        The gravity simulation object.
        """
        if self._simulation is None:
            simulation = self._simulation_class(
                mesh=self.mesh,
                survey=self.survey,
                rhoMap=maps.IdentityMap(nP=self.n_actives),
                ind_active=self.active_cells,
            )
            self._simulation = simulation

        return self._simulation


class SetupExample:
    """
    The example driver.
    """

    _receiver_locations: np.ndarray
    _topography: np.ndarray

    def __init__(
        self,
        driver: type(BaseOctreeSimulationDriver),
        extent: float = 30,
        grid_size: int = 20,
        height: float = 5.0,
        receiver_locations: np.ndarray | None = None,
        topography: np.ndarray | None = None,
        **kwargs,
    ):
        self.extent: float = extent
        self.grid_size: int = grid_size
        self.height: float = height
        self.topography = topography
        self.receiver_locations = receiver_locations

        self.driver = driver(self.receiver_locations, self.topography, **kwargs)

    @property
    def topography(self) -> np.ndarray:
        """
        The topography locations defaulted to a Gaussian surface with twice
        the data extent.
        """
        if self._topography is None:
            extent = self.extent * 2.0

            xr = np.linspace(-extent, extent, self.grid_size * 2)
            yr = np.linspace(-extent, extent, self.grid_size * 2)
            x, y = np.meshgrid(xr, yr)

            # Gaussian surface
            z = -np.exp((x**2 + y**2) / 75**2)
            self._topography = np.c_[utils.mkvc(x.T), utils.mkvc(y.T), utils.mkvc(z.T)]

        return self._topography

    @topography.setter
    def topography(self, topography: np.ndarray | None):
        if not isinstance(topography, (np.ndarray, type(None))):
            raise TypeError(
                f"Attribute 'topography' must be of type np.ndarray, not {type(topography)}."
            )

        self._topography = topography

    @property
    def receiver_locations(self) -> np.ndarray:
        """
        The receiver locations for the simulation.
        """
        if self._receiver_locations is None:
            xy_locations = self.make_grid(self.extent, self.grid_size)
            # Drape the locations on topo
            z_interp = LinearNDInterpolator(
                self.topography[:, :2], self.topography[:, 2]
            )
            elevation = z_interp(xy_locations) + self.height
            self._receiver_locations = np.c_[xy_locations, elevation]

        return self._receiver_locations

    @receiver_locations.setter
    def receiver_locations(self, locations: np.ndarray | None):
        if not isinstance(locations, (np.ndarray, type(None))):
            raise TypeError(
                f"Attribute 'receiver_locations' must be of type np.ndarray, not {type(locations)}."
            )

        self._receiver_locations = locations

    @staticmethod
    def make_grid(extent, resolution) -> np.ndarray:
        """
        Make a grid of positions centered on origin.

        :param extent: Horizontal extent of the grid.
        :param resolution: Number of points along axes of the grid.
        :return: x,y locations of the grid.
        """
        xr = np.linspace(-extent, extent, resolution)
        yr = np.linspace(-extent, extent, resolution)
        x, y = np.meshgrid(xr, yr)

        return np.c_[x.flatten(), y.flatten()]
