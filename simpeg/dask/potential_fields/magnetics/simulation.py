import numpy as np
from ....potential_fields.magnetics import Simulation3DIntegral as Sim
from ....utils import sdiag, mkvc
from dask import array, delayed
from scipy.sparse import csr_matrix as csr


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

        if not self.is_amplitude_data:
            diag = array.einsum("i,ij,ij->j", W, self.Jmatrix, self.Jmatrix)

        else:
            ampDeriv = self.ampDeriv
            J = (
                ampDeriv[0, :, None] * self.Jmatrix[::3]
                + ampDeriv[1, :, None] * self.Jmatrix[1::3]
                + ampDeriv[2, :, None] * self.Jmatrix[2::3]
            )
            diag = ((W[:, None] * J) ** 2).sum(axis=0).compute()
        self._gtg_diagonal = diag
    else:
        diag = self._gtg_diagonal

    mapping_deriv = self.chiDeriv.tocsr().T.power(2)
    dmudm_jtvec = delayed(csr.dot)(mapping_deriv, diag)
    h_vec = array.from_delayed(
        dmudm_jtvec, dtype=np.float32, shape=[self.chiDeriv.shape[1]]
    )

    return h_vec


Sim.getJtJdiag = dask_getJtJdiag
