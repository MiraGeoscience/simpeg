import scipy.sparse as sp
import numpy as np
from simpeg.electromagnetics.natural_source.receivers import Point3DTipper
from simpeg.utils import sdiag


def _eval_tipper_deriv(self, src, mesh, f, du_dm_v=None, v=None, adjoint=False):
    # will grab both primary and secondary and sum them!

    if not isinstance(f, np.ndarray):
        h = f[src, "h"]
    else:
        h = f

    Phx = self.getP(mesh, "Fx", "h")
    Phy = self.getP(mesh, "Fy", "h")
    Phz = self.getP(mesh, "Fz", "h")
    hx = Phx @ h
    hy = Phy @ h
    hz = Phz @ h

    if self.orientation[1] == "x":
        h = -hy
    else:
        h = hx

    top = h[:, 0] * hz[:, 1] - h[:, 1] * hz[:, 0]
    bot = hx[:, 0] * hy[:, 1] - hx[:, 1] * hy[:, 0]
    imp = top / bot

    if adjoint:

        gtop_v = v @ sdiag(bot**-1)
        gbot_v = sdiag(-imp) @ v @ sdiag(bot**-1)

        diag_blocks = gbot_v @ np.c_[hy[:, 1], -hy[:, 0]]
        ghx_v = sp.block_diag(diag_blocks.tolist(), format="csr")

        diag_blocks = gbot_v @ np.c_[-hx[:, 1], hx[:, 0]]
        ghy_v = sp.block_diag(diag_blocks.tolist(), format="csr")

        diag_blocks = gtop_v @ np.c_[-h[:, 1], h[:, 0]]
        ghz_v = sp.block_diag(diag_blocks.tolist(), format="csr")

        diag_blocks = gtop_v @ np.c_[hz[:, 1], -hz[:, 0]]
        gh_v = sp.block_diag(diag_blocks.tolist(), format="csr")

        if self.orientation[1] == "x":
            ghy_v -= gh_v
        else:
            ghx_v += gh_v

        gh_v = Phx.T @ ghx_v + Phy.T @ ghy_v + Phz.T @ ghz_v

        return f._hDeriv(src, None, gh_v, adjoint=True)

    dh_v = f._hDeriv(src, du_dm_v, v, adjoint=False)
    dhx_v = Phx @ dh_v
    dhy_v = Phy @ dh_v
    dhz_v = Phz @ dh_v
    if self.orientation[1] == "x":
        dh_v = -dhy_v
    else:
        dh_v = dhx_v

    dtop_v = (
        h[:, 0] * dhz_v[:, 1]
        + dh_v[:, 0] * hz[:, 1]
        - h[:, 1] * dhz_v[:, 0]
        - dh_v[:, 1] * hz[:, 0]
    )
    dbot_v = (
        hx[:, 0] * dhy_v[:, 1]
        + dhx_v[:, 0] * hy[:, 1]
        - hx[:, 1] * dhy_v[:, 0]
        - dhx_v[:, 1] * hy[:, 0]
    )

    return (bot * dtop_v - top * dbot_v) / (bot * bot)


Point3DTipper._eval_tipper_deriv = _eval_tipper_deriv
