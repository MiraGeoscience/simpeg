import numpy as np
from ....potential_fields.gravity import Simulation3DIntegral as Sim
from ....utils import sdiag, mkvc
from dask import array


def dask_getJtJdiag(self, m, W=None, f=None):
    """
    Return the diagonal of JtJ
    """

    self.model = m
    if getattr(self, "_gtg_diagonal", None) is None:
        if W is None:
            W = np.ones(self.nD)
        else:
            W = W.diagonal()

        diag = array.einsum("i,ij,ij->j", W, self.Jmatrix, self.Jmatrix)

        if isinstance(diag, array.Array):
            diag = np.asarray(diag.compute())

        self._gtg_diagonal = diag
    else:
        diag = self._gtg_diagonal
    return mkvc((sdiag(np.sqrt(diag)) @ self.rhoDeriv).power(2).sum(axis=0))


Sim.getJtJdiag = dask_getJtJdiag
