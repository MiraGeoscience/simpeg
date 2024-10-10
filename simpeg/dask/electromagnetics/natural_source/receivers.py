import scipy.sparse as sp
import numpy as np
from simpeg.electromagnetics.natural_source.receivers import (
    PointNaturalSource,
    Point3DTipper,
)
from simpeg.utils import sdiag


def _eval_impedance_deriv(self, src, mesh, f, du_dm_v=None, v=None, adjoint=False):
    if mesh.dim < 3 and self.orientation in ["xx", "yy"]:
        if adjoint:
            return 0 * v
        else:
            return 0 * du_dm_v
    e = f[src, "e"]
    h = f[src, "h"]
    if mesh.dim == 3:
        if self.orientation[0] == "x":
            Pe = self.getP(mesh, "Ex", "e")
            e = Pe @ e
        else:
            Pe = self.getP(mesh, "Ey", "e")
            e = Pe @ e

        Phx = self.getP(mesh, "Fx", "h")
        Phy = self.getP(mesh, "Fy", "h")
        hx = Phx @ h
        hy = Phy @ h
        if self.orientation[1] == "x":
            h = hy
        else:
            h = -hx

        top = e[:, 0] * h[:, 1] - e[:, 1] * h[:, 0]
        bot = hx[:, 0] * hy[:, 1] - hx[:, 1] * hy[:, 0]
        imp = top / bot
    else:
        if mesh.dim == 1:
            e_loc = f.aliasFields["e"][1]
            h_loc = f.aliasFields["h"][1]
            PE = self.getP(mesh, e_loc)
            PH = self.getP(mesh, h_loc)
        elif mesh.dim == 2:
            if self.orientation == "xy":
                PE = self.getP(mesh, "Ex")
                PH = self.getP(mesh, "CC")
            elif self.orientation == "yx":
                PE = self.getP(mesh, "CC")
                PH = self.getP(mesh, "Ex")

        top = PE @ e[:, 0]
        bot = PH @ h[:, 0]

        if mesh.dim == 1 and self.orientation != f.field_directions:
            bot = -bot

        imp = top / bot

    if adjoint:
        if self.component == "phase":
            # gradient of arctan2(y, x) is (-y/(x**2 + y**2), x/(x**2 + y**2))
            v = 180 / np.pi * imp / (imp.real**2 + imp.imag**2) * v
            # switch real and imaginary, and negate real part of output
            v = -v.imag - 1j * v.real
            # imaginary part gets extra (-) due to conjugate transpose
        elif self.component == "apparent_resistivity":
            v = 2 * _alpha(src) * imp * v
            v = v.real - 1j * v.imag
        elif self.component == "imag":
            v = -1j * v

        # Work backwards!
        gtop_v = sdiag(bot**-1) @ v
        gbot_v = sdiag(-imp / bot) @ v

        if mesh.dim == 3:
            diag_blocks = gbot_v.T @ np.c_[hy[:, 1], -hy[:, 0]]
            ghx_v = sp.block_diag(diag_blocks.tolist(), format="csr")

            diag_blocks = gbot_v.T @ np.c_[-hx[:, 1], hx[:, 0]]
            ghy_v = sp.block_diag(diag_blocks.tolist(), format="csr")

            diag_blocks = gbot_v.T @ np.c_[h[:, 1], -h[:, 0]]
            ge_v = sp.block_diag(diag_blocks.tolist(), format="csr")

            diag_blocks = gbot_v.T @ np.c_[-e[:, 1], e[:, 0]]
            gh_v = sp.block_diag(diag_blocks.tolist(), format="csr")

            if self.orientation[1] == "x":
                ghy_v += gh_v
            else:
                ghx_v -= gh_v

            gh_v = Phx.T @ ghx_v + Phy.T @ ghy_v
            ge_v = Pe.T @ ge_v
        else:
            if mesh.dim == 1 and self.orientation != f.field_directions:
                gbot_v = -gbot_v

            gh_v = PH.T @ gbot_v
            ge_v = PE.T @ gtop_v

        gfu_h_v, gfm_h_v = f._hDeriv(src, None, gh_v, adjoint=True)
        gfu_e_v, gfm_e_v = f._eDeriv(src, None, ge_v, adjoint=True)

        return gfu_h_v + gfu_e_v, gfm_h_v + gfm_e_v

    if mesh.dim == 3:
        de_v = Pe @ f._eDeriv(src, du_dm_v, v, adjoint=False)
        dh_v = f._hDeriv(src, du_dm_v, v, adjoint=False)
        dhx_v = Phx @ dh_v
        dhy_v = Phy @ dh_v
        if self.orientation[1] == "x":
            dh_dm_v = dhy_v
        else:
            dh_dm_v = -dhx_v

        dtop_v = (
            e[:, 0] * dh_dm_v[:, 1]
            + de_v[:, 0] * h[:, 1]
            - e[:, 1] * dh_dm_v[:, 0]
            - de_v[:, 1] * h[:, 0]
        )
        dbot_v = (
            hx[:, 0] * dhy_v[:, 1]
            + dhx_v[:, 0] * hy[:, 1]
            - hx[:, 1] * dhy_v[:, 0]
            - dhx_v[:, 1] * hy[:, 0]
        )
        imp_deriv = (bot * dtop_v - top * dbot_v) / (bot * bot)
    else:
        de_v = PE @ f._eDeriv(src, du_dm_v, v, adjoint=False)
        dh_v = PH @ f._hDeriv(src, du_dm_v, v, adjoint=False)

        if mesh.dim == 1 and self.orientation != f.field_directions:
            dh_v = -dh_v

        imp_deriv = (de_v - imp * dh_v) / bot

    if self.component == "apparent_resistivity":
        rx_deriv = (
            2 * _alpha(src) * (imp.real * imp_deriv.real + imp.imag * imp_deriv.imag)
        )
    elif self.component == "phase":
        amp2 = imp.imag**2 + imp.real**2
        deriv_re = -imp.imag / amp2 * imp_deriv.real
        deriv_im = imp.real / amp2 * imp_deriv.imag

        rx_deriv = (180 / np.pi) * (deriv_re + deriv_im)
    else:
        rx_deriv = getattr(imp_deriv, self.component)
    return rx_deriv


PointNaturalSource._eval_impedance_deriv = _eval_impedance_deriv


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

        gtop_v = sdiag(bot**-1) @ v
        gbot_v = sdiag(-imp / bot) @ v

        diag_blocks = gbot_v.T @ np.c_[hy[:, 1], -hy[:, 0]]
        ghx_v = sp.block_diag(diag_blocks.tolist(), format="csr")

        diag_blocks = gbot_v.T @ np.c_[-hx[:, 1], hx[:, 0]]
        ghy_v = sp.block_diag(diag_blocks.tolist(), format="csr")

        diag_blocks = gtop_v.T @ np.c_[-h[:, 1], h[:, 0]]
        ghz_v = sp.block_diag(diag_blocks.tolist(), format="csr")

        diag_blocks = gtop_v.T @ np.c_[hz[:, 1], -hz[:, 0]]
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
