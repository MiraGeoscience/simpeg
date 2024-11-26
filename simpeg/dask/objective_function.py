from ..objective_function import ComboObjectiveFunction, BaseObjectiveFunction

import dask.array as da

import numpy as np
from dask.distributed import Future, get_client, Client
from ..data_misfit import L2DataMisfit

BaseObjectiveFunction._workers = None


@property
def client(self):
    if getattr(self, "_client", None) is None:
        try:
            self._client = get_client()
        except ValueError:
            self._client = False

    return self._client


@client.setter
def client(self, client):
    assert isinstance(client, Client)
    self._client = client


BaseObjectiveFunction.client = client


@property
def workers(self):
    return self._workers


@workers.setter
def workers(self, workers):
    self._workers = workers


BaseObjectiveFunction.workers = workers


def dask_call(self, m, f=None):
    fcts = []
    multipliers = []
    for i, phi in enumerate(self):
        multiplier, objfct = phi
        if multiplier == 0.0:  # don't evaluate the fct
            continue
        else:

            if self.client and isinstance(objfct, L2DataMisfit):
                fields = f[i] if f is not None else None
                fct = self.client.submit(objfct(m, f=fields))
            else:
                fct = objfct(m)

            fcts += [fct]
            multipliers += [multiplier]

    if self.client and isinstance(fcts[0], Future):
        phi = self.client.gather(fcts)
    else:
        phi = fcts

    value = 0.
    for multiplier, phi in zip(multipliers, phi):
        value += multiplier * phi

    return value


ComboObjectiveFunction.__call__ = dask_call


def dask_deriv(self, m, f=None):
    """
    First derivative of the composite objective function is the sum of the
    derivatives of each objective function in the list, weighted by their
    respective multplier.

    :param numpy.ndarray m: model
    :param SimPEG.Fields f: Fields object (if applicable)
    """

    g = []
    multipliers = []
    for i, phi in enumerate(self):
        multiplier, objfct = phi
        if multiplier == 0.0:  # don't evaluate the fct
            continue
        else:

            if self.client and isinstance(objfct, L2DataMisfit):
                fields = f[i] if f is not None else None
                fct = self.client.submit(objfct.deriv(m, f=fields))
            else:
                fct = objfct.deriv(m)

            g += [fct]
            multipliers += [multiplier]

    if self.client and isinstance(g[0], Future):
        rows = self.client.gather(g)

    deriv = 0.
    for multiplier, g in zip(multipliers, g):
        deriv += multiplier * g

    return deriv


ComboObjectiveFunction.deriv = dask_deriv


def dask_deriv2(self, m, v=None, f=None):
    """
    Second derivative of the composite objective function is the sum of the
    second derivatives of each objective function in the list, weighted by
    their respective multplier.

    :param numpy.ndarray m: model
    :param numpy.ndarray v: vector we are multiplying by
    :param SimPEG.Fields f: Fields object (if applicable)
    """

    H = []
    multipliers = []
    for phi in self:
        multiplier, objfct = phi
        if multiplier == 0.0:  # don't evaluate the fct
            continue
        else:
            if self.client and isinstance(objfct, L2DataMisfit):
                fct = self.client.submit(objfct.deriv2(m, v=v, f=f[i]))
            else:
                fct = objfct.deriv2(m, v=v)

            H += [fct]
            multipliers += [multiplier]

    if self.client and isinstance(H[0], Future):
        H = self.client.gather(H)

    phi_deriv2 = 0
    for multiplier, h in zip(multipliers, H):
        phi_deriv2 += multiplier * h

    return phi_deriv2


ComboObjectiveFunction.deriv2 = dask_deriv2


def getJtJdiag(self, m):

    jtj_diags = []
    multipliers = []
    for multiplier, dmisfit in self:

        if self.client:
            jtj_diags.append(self.client.persist(dmisfit.getJtJdiag(m), pure=False))
        else:
            jtj_diags.append(dmisfit.getJtJdiag(m))

        multipliers += [multiplier]

    if self.client:
        result = self.client.gather(jtj_diags)
    else:
        result = jtj_diags


    jtj_diag = 0.
    for multiplier, row in zip(multipliers, jtj_diags):
        jtj_diag += multiplier * row

    return np.asarray(jtj_diag)


ComboObjectiveFunction.getJtJdiag = getJtJdiag
