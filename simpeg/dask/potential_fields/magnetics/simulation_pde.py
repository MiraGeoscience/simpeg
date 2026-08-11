from dask import array, compute, delayed
import numpy as np
from ....potential_fields.magnetics import Simulation3DDifferential as Sim
from ....utils import sdiag, mkvc


def distance_weights(
    indices,
    locations,
    uncertainties,
    cell_centers,
    cell_volumes,
    exponent=3,
    threshold=1e-2,
):
    weights = np.zeros(len(cell_centers))
    for ind in indices:
        distance = np.linalg.norm(cell_centers - locations[ind], axis=1)
        weights += (
            uncertainties[ind] ** 2.0
            * cell_volumes**2.0
            * (distance + threshold) ** (-2 * exponent)
        )

    return weights


def dask_getJtJdiag(self, m, W=None, f=None):
    """
    Return the diagonal of JtJ
    """

    self.model = m
    print("In new getJtJdiag")
    if W is None:
        uncertainties = np.ones(self.Jmatrix.shape[0])
    else:
        uncertainties = W.diagonal()

    if getattr(self, "_jtjdiag", None) is None:
        client, worker = self._get_client_worker()

        n_threads = self.n_threads(client=client, worker=worker) // 2

        chunks = np.array_split(
            np.arange(self.survey.receiver_locations.shape[0]), n_threads
        )
        cell_centers = self.mesh.cell_centers.copy()
        cell_volumes = self.mesh.cell_volumes.copy()
        locations = self.survey.receiver_locations.copy()

        if client:
            cell_centers = client.scatter(cell_centers, workers=worker)
            cell_volumes = client.scatter(cell_volumes, workers=worker)
            locations = client.scatter(locations, workers=worker)
            uncertainties = client.scatter(uncertainties, workers=worker)
        else:
            delayed_distance_weights = delayed(distance_weights)

        futures = []
        for block in chunks:
            if client:
                futures.append(
                    client.submit(
                        distance_weights,
                        block,
                        locations,
                        uncertainties,
                        cell_centers,
                        cell_volumes,
                        workers=worker,
                    )
                )
            else:
                futures.append(
                    array.from_delayed(
                        delayed_distance_weights(
                            block,
                            locations,
                            uncertainties,
                            cell_centers,
                            cell_volumes,
                        ),
                        dtype=np.float32,
                        shape=(
                            len(block),
                            len(cell_centers),
                        ),
                    )
                )

        if client:
            diag = client.gather(futures)
        else:
            diag = compute(futures)[0]

        diag = np.tile(np.vstack(diag).sum(axis=0), 3)

        self._jtjdiag = diag

    return mkvc((sdiag(np.sqrt(self._jtjdiag)) @ self.remDeriv).power(2).sum(axis=0))


Sim.getJtJdiag = dask_getJtJdiag
