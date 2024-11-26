import numpy as np

from ..data_misfit import L2DataMisfit
from ..fields import Fields
from ..utils import mkvc
from .utils import compute
import dask.array as da
from scipy.sparse import csr_matrix as csr
from dask import array, delayed


def dask_dpred(self, m, compute_J=False):
    """
    Distributed :obj:`simpeg.data_misfit.L2DataMisfit.dpred`
    """
    mapping_deriv = self.model_map.deriv(m)
    if getattr(self, "model_map", None) is not None:
        m = self.model_map @ m

    return self.simulation.dpred(m, compute_J=compute_J)

L2DataMisfit.dpred = dask_dpred

def dask_call(self, m, f=None):
    """
    Distributed :obj:`simpeg.data_misfit.L2DataMisfit.__call__`
    """
    R = self.W * self.residual(m, f=f)
    phi_d = da.dot(R, R)
    if not isinstance(phi_d, np.ndarray):
        return self.phi_d
    return phi_d


L2DataMisfit.__call__ = dask_call


def dask_deriv(self, m, f=None):
    """
    Distributed :obj:`simpeg.data_misfit.L2DataMisfit.deriv`
    """
    mapping_deriv = self.model_map.deriv(m)
    if getattr(self, "model_map", None) is not None:
        m = self.model_map @ m

    wtw_d = self.W.diagonal() ** 2.0 * self.residual(m, f=f)
    Jtvec = self.simulation.Jtvec(m, wtw_d)

    if getattr(self, "model_map", None) is not None:
        Jtjvec_dmudm = delayed(csr.dot)(Jtvec, mapping_deriv)
        h_vec = da.from_delayed(
            Jtjvec_dmudm, dtype=float, shape=[mapping_deriv.shape[1]]
        )
        return h_vec

    return Jtvec


L2DataMisfit.deriv = dask_deriv


def dask_deriv2(self, m, v, f=None):
    """
    Distributed :obj:`simpeg.data_misfit.L2DataMisfit.deriv2`
    """
    mapping_deriv = self.model_map.deriv(m)
    if getattr(self, "model_map", None) is not None:
        m = self.model_map @ m
        v = mapping_deriv @ v

    jvec = self.simulation.Jvec(m, v)
    w_jvec = self.W.diagonal() ** 2.0 * jvec
    jtwjvec = self.simulation.Jtvec(m, w_jvec)

    if getattr(self, "model_map", None) is not None:
        Jtjvec_dmudm = delayed(csr.dot)(jtwjvec, mapping_deriv)
        h_vec = da.from_delayed(
            Jtjvec_dmudm, dtype=float, shape=[mapping_deriv.shape[1]]
        )
        return h_vec

    return jtwjvec


L2DataMisfit.deriv2 = dask_deriv2


def dask_residual(self, m, f=None):
    if self.data is None:
        raise Exception("data must be set before a residual can be calculated.")

    if isinstance(f, Fields) or f is None:
        return self.simulation.residual(m, self.data.dobs, f=f)
    elif f.shape == self.data.dobs.shape:
        return mkvc(f - self.data.dobs)
    else:
        raise Exception(f"Attribute f must be or type {Fields}, numpy.array or None.")


L2DataMisfit.residual = dask_residual


def getJtJdiag(self, m):
    """
    Evaluate the main diagonal of JtJ
    """
    if getattr(self.simulation, "getJtJdiag", None) is None:
        raise AttributeError(
            "Simulation does not have a getJtJdiag attribute."
            + "Cannot form the sensitivity explicitly"
        )

    if self.model_map is not None:
        mapping_deriv = self.model_map.deriv(m)
        m = mapping_deriv @ m

    jtjdiag = self.simulation.getJtJdiag(m, W=self.W)

    if self.model_map is not None:
        mapping_deriv = self.model_map.deriv(m).tocsr().T.power(2)
        dmudm_jtvec = delayed(csr.dot)(mapping_deriv, jtjdiag)
        jtjdiag = array.from_delayed(
            dmudm_jtvec, dtype=np.float32, shape=[mapping_deriv.shape[1]]
        )

    return jtjdiag

L2DataMisfit.getJtJdiag = getJtJdiag