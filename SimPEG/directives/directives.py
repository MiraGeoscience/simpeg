import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
import scipy.sparse as sp
from ..data_misfit import BaseDataMisfit
from ..objective_function import ComboObjectiveFunction
from ..maps import IdentityMap, Wires
from ..regularization import (
    WeightedLeastSquares,
    BaseRegularization,
    BaseSparse,
    Smallness,
    Sparse,
    SparseSmallness,
    PGIsmallness,
    PGIwithNonlinearRelationshipsSmallness,
    SmoothnessFirstOrder,
    SparseSmoothness,
    BaseSimilarityMeasure,
)
from ..utils import (
    mkvc,
    set_kwargs,
    sdiag,
    diagEst,
    spherical2cartesian,
    cartesian2spherical,
    Zero,
    eigenvalue_by_power_iteration,
    validate_string,
)
from ..utils.code_utils import (
    deprecate_property,
    validate_type,
    validate_integer,
    validate_float,
    validate_ndarray_with_shape,
)
from .. import optimization


class InversionDirective:
    """InversionDirective"""

    _REGISTRY = {}

    _regPair = [WeightedLeastSquares, BaseRegularization, ComboObjectiveFunction]
    _dmisfitPair = [BaseDataMisfit, ComboObjectiveFunction]

    def __init__(self, inversion=None, dmisfit=None, reg=None, verbose=False, **kwargs):
        self.inversion = inversion
        self.dmisfit = dmisfit
        self.reg = reg
        debug = kwargs.pop("debug", None)
        if debug is not None:
            self.debug = debug
        else:
            self.verbose = verbose
        set_kwargs(self, **kwargs)

    @property
    def verbose(self):
        """Whether to print debug information.

        Returns
        -------
        bool
        """
        return self._verbose

    @verbose.setter
    def verbose(self, value):
        self._verbose = validate_type("verbose", value, bool)

    debug = deprecate_property(verbose, "debug", "verbose", removal_version="0.19.0")

    @property
    def inversion(self):
        """This is the inversion of the InversionDirective instance."""
        if not hasattr(self, "_inversion"):
            return None
        return self._inversion

    @inversion.setter
    def inversion(self, i):
        if getattr(self, "_inversion", None) is not None:
            warnings.warn(
                "InversionDirective {0!s} has switched to a new inversion.".format(
                    self.__class__.__name__
                )
            )
        self._inversion = i

    @property
    def invProb(self):
        return self.inversion.invProb

    @property
    def opt(self):
        return self.invProb.opt

    @property
    def reg(self):
        if getattr(self, "_reg", None) is None:
            self.reg = self.invProb.reg  # go through the setter
        return self._reg

    @reg.setter
    def reg(self, value):
        if value is not None:
            assert any(
                [isinstance(value, regtype) for regtype in self._regPair]
            ), "Regularization must be in {}, not {}".format(self._regPair, type(value))

            if isinstance(value, WeightedLeastSquares):
                value = 1 * value  # turn it into a combo objective function
        self._reg = value

    @property
    def dmisfit(self):
        if getattr(self, "_dmisfit", None) is None:
            self.dmisfit = self.invProb.dmisfit  # go through the setter
        return self._dmisfit

    @dmisfit.setter
    def dmisfit(self, value):
        if value is not None:

            assert any(
                [isinstance(value, dmisfittype) for dmisfittype in self._dmisfitPair]
            ), "Misfit must be in {}, not {}".format(self._dmisfitPair, type(value))

            if not isinstance(value, ComboObjectiveFunction):
                value = 1 * value  # turn it into a combo objective function
        self._dmisfit = value

    @property
    def survey(self):
        """
        Assuming that dmisfit is always a ComboObjectiveFunction,
        return a list of surveys for each dmisfit [survey1, survey2, ... ]
        """
        return [objfcts.simulation.survey for objfcts in self.dmisfit.objfcts]

    @property
    def simulation(self):
        """
        Assuming that dmisfit is always a ComboObjectiveFunction,
        return a list of problems for each dmisfit [prob1, prob2, ...]
        """
        return [objfcts.simulation for objfcts in self.dmisfit.objfcts]

    def initialize(self):
        pass

    def endIter(self):
        pass

    def finish(self):
        pass

    def validate(self, directiveList=None):
        return True


class DirectiveList(object):
    def __init__(self, *directives, inversion=None, debug=False, **kwargs):
        super().__init__(**kwargs)
        self.dList = []
        for d in directives:
            assert isinstance(
                d, InversionDirective
            ), "All directives must be InversionDirectives not {}".format(type(d))
            self.dList.append(d)

        self.inversion = inversion
        self.verbose = debug

    @property
    def debug(self):
        return getattr(self, "_debug", False)

    @debug.setter
    def debug(self, value):
        for d in self.dList:
            d.debug = value
        self._debug = value

    @property
    def inversion(self):
        """This is the inversion of the InversionDirective instance."""
        return getattr(self, "_inversion", None)

    @inversion.setter
    def inversion(self, i):
        if self.inversion is i:
            return
        if getattr(self, "_inversion", None) is not None:
            warnings.warn(
                "{0!s} has switched to a new inversion.".format(self.__class__.__name__)
            )
        for d in self.dList:
            d.inversion = i
        self._inversion = i

    def call(self, ruleType):
        if self.dList is None:
            if self.verbose:
                print("DirectiveList is None, no directives to call!")
            return

        directives = ["initialize", "endIter", "finish"]
        assert ruleType in directives, 'Directive type must be in ["{0!s}"]'.format(
            '", "'.join(directives)
        )
        for r in self.dList:
            getattr(r, ruleType)()

    def validate(self):
        [directive.validate(self) for directive in self.dList]
        return True


class BetaEstimate_ByEig(InversionDirective):
    """
    Estimate the trade-off parameter beta between the data misfit(s) and the
    regularization as a multiple of the ratio between the highest eigenvalue of the
    data misfit term and the highest eigenvalue of the regularization.
    The highest eigenvalues are estimated through power iterations and Rayleigh quotient.

    """

    def __init__(self, beta0_ratio=1.0, n_pw_iter=4, seed=None, method="power_iteration", **kwargs):
        super().__init__(**kwargs)
        self.method = method
        self.beta0_ratio = beta0_ratio
        self.n_pw_iter = n_pw_iter
        self.seed = seed

    @property
    def beta0_ratio(self):
        """The estimated ratio is multiplied by this to obtain beta.

        Returns
        -------
        float
        """
        return self._beta0_ratio

    @beta0_ratio.setter
    def beta0_ratio(self, value):
        self._beta0_ratio = validate_float(
            "beta0_ratio", value, min_val=0.0, inclusive_min=False
        )

    @property
    def n_pw_iter(self):
        """Number of power iterations for estimation.

        Returns
        -------
        int
        """
        return self._n_pw_iter

    @n_pw_iter.setter
    def n_pw_iter(self, value):
        self._n_pw_iter = validate_integer("n_pw_iter", value, min_val=1)

    @property
    def seed(self):
        """Random seed to initialize with

        Returns
        -------
        int
        """
        return self._seed

    @seed.setter
    def seed(self, value):
        if value is not None:
            value = validate_integer("seed", value, min_val=1)
        self._seed = value

    def initialize(self):
        r"""
        The initial beta is calculated by comparing the estimated
        eigenvalues of JtJ and WtW.
        To estimate the eigenvector of **A**, we will use one iteration
        of the *Power Method*:

        .. math::
            \mathbf{x_1 = A x_0}

        Given this (very course) approximation of the eigenvector, we can
        use the *Rayleigh quotient* to approximate the largest eigenvalue.

        .. math::
            \lambda_0 = \frac{\mathbf{x^\top A x}}{\mathbf{x^\top x}}

        We will approximate the largest eigenvalue for both JtJ and WtW,
        and use some ratio of the quotient to estimate beta0.

        .. math::
            \beta_0 =
                \gamma
                \frac{
                    \mathbf{x^\top J^\top J x}
                }{
                    \mathbf{x^\top W^\top W x}
                }

        :rtype: float
        :return: beta0
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        if self.verbose:
            print("Calculating the beta0 parameter.")

        m = self.invProb.model

        if self.method == "power_iteration":
            dm_eigenvalue = eigenvalue_by_power_iteration(
                self.dmisfit, m, n_pw_iter=self.n_pw_iter,
            )
            reg_eigenvalue = eigenvalue_by_power_iteration(
                self.reg, m, n_pw_iter=self.n_pw_iter,
            )
            self.ratio = np.asarray(dm_eigenvalue / reg_eigenvalue)
        else:
            x0 = np.random.rand(*m.shape)
            phi_d_deriv = self.dmisfit.deriv2(m, x0)
            t = np.dot(x0, phi_d_deriv)
            reg = self.reg.deriv2(m, v=x0)
            b = np.dot(x0, reg)
            self.ratio = np.asarray(t / b)

        self.beta0 = self.beta0_ratio * self.ratio

        self.invProb.beta = self.beta0


class BetaSchedule(InversionDirective):
    """BetaSchedule"""

    def __init__(self, coolingFactor=8.0, coolingRate=3, **kwargs):
        super().__init__(**kwargs)
        self.coolingFactor = coolingFactor
        self.coolingRate = coolingRate

    @property
    def coolingFactor(self):
        """Beta is divided by this value every `coolingRate` iterations.

        Returns
        -------
        float
        """
        return self._coolingFactor

    @coolingFactor.setter
    def coolingFactor(self, value):
        self._coolingFactor = validate_float(
            "coolingFactor", value, min_val=0.0, inclusive_min=False
        )

    @property
    def coolingRate(self):
        """Cool after this number of iterations.

        Returns
        -------
        int
        """
        return self._coolingRate

    @coolingRate.setter
    def coolingRate(self, value):
        self._coolingRate = validate_integer("coolingRate", value, min_val=1)

    def endIter(self):
        if self.opt.iter > 0 and self.opt.iter % self.coolingRate == 0:
            if self.verbose:
                print(
                    "BetaSchedule is cooling Beta. Iteration: {0:d}".format(
                        self.opt.iter
                    )
                )
            self.invProb.beta /= self.coolingFactor


class AlphasSmoothEstimate_ByEig(InversionDirective):
    """
    Estimate the alphas multipliers for the smoothness terms of the regularization
    as a multiple of the ratio between the highest eigenvalue of the
    smallness term and the highest eigenvalue of each smoothness term of the regularization.
    The highest eigenvalue are estimated through power iterations and Rayleigh quotient.
    """

    def __init__(self, alpha0_ratio=1.0, n_pw_iter=4, seed=None, **kwargs):
        super().__init__(**kwargs)
        self.alpha0_ratio = alpha0_ratio
        self.n_pw_iter = n_pw_iter
        self.seed = seed

    @property
    def alpha0_ratio(self):
        """the estimated Alpha_smooth is multiplied by this ratio (int or array)

        Returns
        -------
        numpy.ndarray
        """
        return self._alpha0_ratio

    @alpha0_ratio.setter
    def alpha0_ratio(self, value):
        self._alpha0_ratio = validate_ndarray_with_shape(
            "alpha0_ratio", value, shape=("*",)
        )

    @property
    def n_pw_iter(self):
        """Number of power iterations for estimation.

        Returns
        -------
        int
        """
        return self._n_pw_iter

    @n_pw_iter.setter
    def n_pw_iter(self, value):
        self._n_pw_iter = validate_integer("n_pw_iter", value, min_val=1)

    @property
    def seed(self):
        """Random seed to initialize with

        Returns
        -------
        int
        """
        return self._seed

    @seed.setter
    def seed(self, value):
        if value is not None:
            value = validate_integer("seed", value, min_val=1)
        self._seed = value

    def initialize(self):
        """"""
        if self.seed is not None:
            np.random.seed(self.seed)

        smoothness = []
        smallness = []
        parents = {}
        for regobjcts in self.reg.objfcts:

            if isinstance(regobjcts, ComboObjectiveFunction):
                objfcts = regobjcts.objfcts
            else:
                objfcts = [regobjcts]

            for obj in objfcts:

                if isinstance(
                    obj,
                    (
                        Smallness,
                        SparseSmallness,
                        PGIsmallness,
                        PGIwithNonlinearRelationshipsSmallness,
                    ),
                ):
                    smallness += [obj]

                elif isinstance(obj, (SmoothnessFirstOrder, SparseSmoothness)):
                    parents[obj] = regobjcts
                    smoothness += [obj]

        if len(smallness) == 0:
            raise UserWarning(
                "Directive 'AlphasSmoothEstimate_ByEig' requires a regularization with at least one Small instance."
            )

        smallness_eigenvalue = eigenvalue_by_power_iteration(
            smallness[0],
            self.invProb.model,
            n_pw_iter=self.n_pw_iter,
        )

        self.alpha0_ratio = self.alpha0_ratio * np.ones(len(smoothness))

        if len(self.alpha0_ratio) != len(smoothness):
            raise ValueError(
                f"Input values for 'alpha0_ratio' should be of len({len(smoothness)}). Provided {self.alpha0_ratio}"
            )

        alphas = []
        for user_alpha, obj in zip(self.alpha0_ratio, smoothness):
            smooth_i_eigenvalue = eigenvalue_by_power_iteration(
                obj,
                self.invProb.model,
                n_pw_iter=self.n_pw_iter,
            )
            ratio = smallness_eigenvalue / smooth_i_eigenvalue

            mtype = obj._multiplier_pair

            new_alpha = getattr(parents[obj], mtype) * user_alpha * ratio
            setattr(parents[obj], mtype, new_alpha)
            alphas += [new_alpha]

        if self.verbose:
            print(f"Alpha scales: {alphas}")


class ScalingMultipleDataMisfits_ByEig(InversionDirective):
    """
    For multiple data misfits only: multiply each data misfit term
    by the inverse of its highest eigenvalue and then
    normalize the sum of the data misfit multipliers to one.
    The highest eigenvalue are estimated through power iterations and Rayleigh quotient.
    """

    def __init__(self, chi0_ratio=None, n_pw_iter=4, seed=None, **kwargs):
        super().__init__(**kwargs)
        self.chi0_ratio = chi0_ratio
        self.n_pw_iter = n_pw_iter
        self.seed = seed

    @property
    def chi0_ratio(self):
        """the estimated Alpha_smooth is multiplied by this ratio (int or array)

        Returns
        -------
        numpy.ndarray
        """
        return self._chi0_ratio

    @chi0_ratio.setter
    def chi0_ratio(self, value):
        if value is not None:
            value = validate_ndarray_with_shape("chi0_ratio", value, shape=("*",))
        self._chi0_ratio = value

    @property
    def n_pw_iter(self):
        """Number of power iterations for estimation.

        Returns
        -------
        int
        """
        return self._n_pw_iter

    @n_pw_iter.setter
    def n_pw_iter(self, value):
        self._n_pw_iter = validate_integer("n_pw_iter", value, min_val=1)

    @property
    def seed(self):
        """Random seed to initialize with

        Returns
        -------
        int
        """
        return self._seed

    @seed.setter
    def seed(self, value):
        if value is not None:
            value = validate_integer("seed", value, min_val=1)
        self._seed = value

    def initialize(self):
        """"""
        if self.seed is not None:
            np.random.seed(self.seed)

        if self.verbose:
            print("Calculating the scaling parameter.")

        if (
            getattr(self.dmisfit, "objfcts", None) is None
            or len(self.dmisfit.objfcts) == 1
        ):
            raise TypeError(
                "ScalingMultipleDataMisfits_ByEig only applies to joint inversion"
            )

        ndm = len(self.dmisfit.objfcts)
        if self.chi0_ratio is not None:
            self.chi0_ratio = self.chi0_ratio * np.ones(ndm)
        else:
            self.chi0_ratio = self.dmisfit.multipliers

        m = self.invProb.model

        dm_eigenvalue_list = []
        for j, dm in enumerate(self.dmisfit.objfcts):
            dm_eigenvalue_list += [eigenvalue_by_power_iteration(dm, m)]

        self.chi0 = self.chi0_ratio / np.r_[dm_eigenvalue_list]
        self.chi0 = self.chi0 / np.sum(self.chi0)
        self.dmisfit.multipliers = self.chi0

        if self.verbose:
            print("Scale Multipliers: ", self.dmisfit.multipliers)


class JointScalingSchedule(InversionDirective):
    """
    For multiple data misfits only: rebalance each data misfit term
    during the inversion when some datasets are fit, and others not
    using the ratios of current misfits and their respective target.
    It implements the strategy described in https://doi.org/10.1093/gji/ggaa378.
    """

    def __init__(
        self, warmingFactor=1.0, chimax=1e10, chimin=1e-10, update_rate=1, **kwargs
    ):
        super().__init__(**kwargs)
        self.mode = 1
        self.warmingFactor = warmingFactor
        self.chimax = chimax
        self.chimin = chimin
        self.update_rate = update_rate

    @property
    def mode(self):
        """The type of update to perform.

        Returns
        -------
        {1, 2}
        """
        return self._mode

    @mode.setter
    def mode(self, value):
        self._mode = validate_integer("mode", value, min_val=1, max_val=2)

    @property
    def warmingFactor(self):
        """Factor to adjust scaling of the data misfits by.

        Returns
        -------
        float
        """
        return self._warmingFactor

    @warmingFactor.setter
    def warmingFactor(self, value):
        self._warmingFactor = validate_float(
            "warmingFactor", value, min_val=0.0, inclusive_min=False
        )

    @property
    def chimax(self):
        """Maximum chi factor.

        Returns
        -------
        float
        """
        return self._chimax

    @chimax.setter
    def chimax(self, value):
        self._chimax = validate_float("chimax", value, min_val=0.0, inclusive_min=False)

    @property
    def chimin(self):
        """Minimum chi factor.

        Returns
        -------
        float
        """
        return self._chimin

    @chimin.setter
    def chimin(self, value):
        self._chimin = validate_float("chimin", value, min_val=0.0, inclusive_min=False)

    @property
    def update_rate(self):
        """Will update the data misfit scalings after this many iterations.

        Returns
        -------
        int
        """
        return self._update_rate

    @update_rate.setter
    def update_rate(self, value):
        self._update_rate = validate_integer("update_rate", value, min_val=1)

    def initialize(self):

        if (
            getattr(self.dmisfit, "objfcts", None) is None
            or len(self.dmisfit.objfcts) == 1
        ):
            raise TypeError("JointScalingSchedule only applies to joint inversion")

        targetclass = np.r_[
            [
                isinstance(dirpart, MultiTargetMisfits)
                for dirpart in self.inversion.directiveList.dList
            ]
        ]
        if ~np.any(targetclass):
            self.DMtarget = None
        else:
            self.targetclass = np.where(targetclass)[0][-1]
            self.DMtarget = self.inversion.directiveList.dList[
                self.targetclass
            ].DMtarget

        if self.verbose:
            print("Initial data misfit scales: ", self.dmisfit.multipliers)

    def endIter(self):

        self.dmlist = self.inversion.directiveList.dList[self.targetclass].dmlist

        if np.any(self.dmlist < self.DMtarget):
            self.mode = 2
        else:
            self.mode = 1

        if self.opt.iter > 0 and self.opt.iter % self.update_rate == 0:

            if self.mode == 2:

                if np.all(np.r_[self.dmisfit.multipliers] > self.chimin) and np.all(
                    np.r_[self.dmisfit.multipliers] < self.chimax
                ):

                    indx = self.dmlist > self.DMtarget
                    if np.any(indx):
                        multipliers = self.warmingFactor * np.median(
                            self.DMtarget[~indx] / self.dmlist[~indx]
                        )
                        if np.sum(indx) == 1:
                            indx = np.where(indx)[0][0]
                        self.dmisfit.multipliers[indx] *= multipliers
                        self.dmisfit.multipliers /= np.sum(self.dmisfit.multipliers)

                        if self.verbose:
                            print("Updating scaling for data misfits by ", multipliers)
                            print("New scales:", self.dmisfit.multipliers)


class TargetMisfit(InversionDirective):
    """
    ... note:: Currently this target misfit is not set up for joint inversion.
    Check out MultiTargetMisfits
    """

    def __init__(self, target=None, phi_d_star=None, chifact=1.0, **kwargs):
        super().__init__(**kwargs)
        self.chifact = chifact
        self.phi_d_star = phi_d_star
        if phi_d_star is not None and target is not None:
            raise AttributeError("Attempted to set both target and phi_d_star.")
        if target is not None:
            self.target = target

    @property
    def target(self):
        """The target value for the data misfit

        Returns
        -------
        float
        """
        if getattr(self, "_target", None) is None:
            self._target = self.chifact * self.phi_d_star
        return self._target

    @target.setter
    def target(self, val):
        self._target = validate_float("target", val, min_val=0.0, inclusive_min=False)

    @property
    def chifact(self):
        """The a multiplier for the target data misfit value.

        The target value is `chifact` times `phi_d_star`

        Returns
        -------
        float
        """
        return self._chifact

    @chifact.setter
    def chifact(self, value):
        self._chifact = validate_float(
            "chifact", value, min_val=0.0, inclusive_min=False
        )
        self._target = None

    @property
    def phi_d_star(self):
        """The target phi_d value for the data misfit.

        The target value is `chifact` times `phi_d_star`

        Returns
        -------
        float
        """
        # the factor of 0.5 is because we do phid = 0.5*||dpred - dobs||^2
        if self._phi_d_star is None:
            nD = 0
            for survey in self.survey:
                nD += survey.nD
            self._phi_d_star = 0.5 * nD
        return self._phi_d_star

    @phi_d_star.setter
    def phi_d_star(self, value):
        # the factor of 0.5 is because we do phid = 0.5*||dpred - dobs||^2
        if value is not None:
            value = validate_float(
                "phi_d_star", value, min_val=0.0, inclusive_min=False
            )
        self._phi_d_star = value
        self._target = None

    def endIter(self):
        if self.invProb.phi_d < self.target:
            self.opt.stopNextIteration = True
            self.print_final_misfit()

    def print_final_misfit(self):
        if self.opt.print_type == "ubc":
            self.opt.print_target = (
                ">> Target misfit: %.1f (# of data) is achieved"
            ) % (self.target * self.invProb.opt.factor)


class MultiTargetMisfits(InversionDirective):
    def __init__(
        self,
        WeightsInTarget=False,
        chifact=1.0,
        phi_d_star=None,
        TriggerSmall=True,
        chiSmall=1.0,
        phi_ms_star=None,
        TriggerTheta=False,
        ToleranceTheta=1.0,
        distance_norm=np.inf,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.WeightsInTarget = WeightsInTarget
        # Chi factor for Geophsyical Data Misfit
        self.chifact = chifact
        self.phi_d_star = phi_d_star

        # Chifact for Clustering/Smallness
        self.TriggerSmall = TriggerSmall
        self.chiSmall = chiSmall
        self.phi_ms_star = phi_ms_star

        # Tolerance for parameters difference with their priors
        self.TriggerTheta = TriggerTheta  # deactivated by default
        self.ToleranceTheta = ToleranceTheta
        self.distance_norm = distance_norm

        self._DM = False
        self._CL = False
        self._DP = False

    @property
    def WeightsInTarget(self):
        """Whether to account for weights in the petrophysical misfit.

        Returns
        -------
        bool
        """
        return self._WeightsInTarget

    @WeightsInTarget.setter
    def WeightsInTarget(self, value):
        self._WeightsInTarget = validate_type("WeightsInTarget", value, bool)

    @property
    def chifact(self):
        """The a multiplier for the target Geophysical data misfit value.

        The target value is `chifact` times `phi_d_star`

        Returns
        -------
        numpy.ndarray
        """
        return self._chifact

    @chifact.setter
    def chifact(self, value):
        self._chifact = validate_ndarray_with_shape("chifact", value, shape=("*",))
        self._DMtarget = None

    @property
    def phi_d_star(self):
        """The target phi_d value for the Geophysical data misfit.

        The target value is `chifact` times `phi_d_star`

        Returns
        -------
        float
        """
        # the factor of 0.5 is because we do phid = 0.5*|| dpred - dobs||^2
        if getattr(self, "_phi_d_star", None) is None:
            # Check if it is a ComboObjective
            if isinstance(self.dmisfit, ComboObjectiveFunction):
                value = np.r_[[0.5 * survey.nD for survey in self.survey]]
            else:
                value = np.r_[[0.5 * self.survey.nD]]
            self._phi_d_star = value
            self._DMtarget = None

        return self._phi_d_star

    @phi_d_star.setter
    def phi_d_star(self, value):
        # the factor of 0.5 is because we do phid = 0.5*|| dpred - dobs||^2
        if value is not None:
            value = validate_ndarray_with_shape("phi_d_star", value, shape=("*",))
        self._phi_d_star = value
        self._DMtarget = None

    @property
    def chiSmall(self):
        """The a multiplier for the target petrophysical misfit value.

        The target value is `chiSmall` times `phi_ms_star`

        Returns
        -------
        float
        """
        return self._chiSmall

    @chiSmall.setter
    def chiSmall(self, value):
        self._chiSmall = validate_float("chiSmall", value)
        self._CLtarget = None

    @property
    def phi_ms_star(self):
        """The target value for the petrophysical data misfit.

        The target value is `chiSmall` times `phi_ms_star`

        Returns
        -------
        float
        """
        return self._phi_ms_star

    @phi_ms_star.setter
    def phi_ms_star(self, value):
        if value is not None:
            value = validate_float("phi_ms_star", value)
        self._phi_ms_star = value
        self._CLtarget = None

    @property
    def TriggerSmall(self):
        """Whether to trigger the smallness misfit test.

        Returns
        -------
        bool
        """
        return self._TriggerSmall

    @TriggerSmall.setter
    def TriggerSmall(self, value):
        self._TriggerSmall = validate_type("TriggerSmall", value, bool)

    @property
    def TriggerTheta(self):
        """Whether to trigger the GMM misfit test.

        Returns
        -------
        bool
        """
        return self._TriggerTheta

    @TriggerTheta.setter
    def TriggerTheta(self, value):
        self._TriggerTheta = validate_type("TriggerTheta", value, bool)

    @property
    def ToleranceTheta(self):
        """Target value for the GMM misfit.

        Returns
        -------
        float
        """
        return self._ToleranceTheta

    @ToleranceTheta.setter
    def ToleranceTheta(self, value):
        self._ToleranceTheta = validate_float("ToleranceTheta", value, min_val=0.0)

    @property
    def distance_norm(self):
        """Distance norm to use for GMM misfit measure.

        Returns
        -------
        float
        """
        return self._distance_norm

    @distance_norm.setter
    def distance_norm(self, value):
        self._distance_norm = validate_float("distance_norm", value, min_val=0.0)

    def initialize(self):
        self.dmlist = np.r_[[dmis(self.invProb.model) for dmis in self.dmisfit.objfcts]]

        if getattr(self.invProb.reg.objfcts[0], "objfcts", None) is not None:
            smallness = np.r_[
                [
                    (
                        np.r_[
                            i,
                            j,
                            (
                                isinstance(
                                    regpart,
                                    PGIwithNonlinearRelationshipsSmallness,
                                )
                                or isinstance(regpart, PGIsmallness)
                            ),
                        ]
                    )
                    for i, regobjcts in enumerate(self.invProb.reg.objfcts)
                    for j, regpart in enumerate(regobjcts.objfcts)
                ]
            ]
            if smallness[smallness[:, 2] == 1][:, :2].size == 0:
                warnings.warn(
                    "There is no PGI regularization. Smallness target is turned off (TriggerSmall flag)"
                )
                self.smallness = -1
                self.pgi_smallness = None

            else:
                self.smallness = smallness[smallness[:, 2] == 1][:, :2][0]
                self.pgi_smallness = self.invProb.reg.objfcts[
                    self.smallness[0]
                ].objfcts[self.smallness[1]]

                if self.verbose:
                    print(
                        type(
                            self.invProb.reg.objfcts[self.smallness[0]].objfcts[
                                self.smallness[1]
                            ]
                        )
                    )

            self._regmode = 1

        else:
            smallness = np.r_[
                [
                    (
                        np.r_[
                            j,
                            (
                                isinstance(
                                    regpart,
                                    PGIwithNonlinearRelationshipsSmallness,
                                )
                                or isinstance(regpart, PGIsmallness)
                            ),
                        ]
                    )
                    for j, regpart in enumerate(self.invProb.reg.objfcts)
                ]
            ]
            if smallness[smallness[:, 1] == 1][:, :1].size == 0:
                if self.TriggerSmall:
                    warnings.warn(
                        "There is no PGI regularization. Smallness target is turned off (TriggerSmall flag)."
                    )
                    self.TriggerSmall = False
                self.smallness = -1
            else:
                self.smallness = smallness[smallness[:, 1] == 1][:, :1][0]
                self.pgi_smallness = self.invProb.reg.objfcts[self.smallness[0]]

                if self.verbose:
                    print(type(self.invProb.reg.objfcts[self.smallness[0]]))

            self._regmode = 2

    @property
    def DM(self):
        """Whether the geophysical data misfit target was satisfied.

        Returns
        -------
        bool
        """
        return self._DM

    @property
    def CL(self):
        """Whether the petrophysical misfit target was satisified.

        Returns
        -------
        bool
        """
        return self._CL

    @property
    def DP(self):
        """Whether the GMM misfit was below the threshold.

        Returns
        -------
        bool
        """
        return self._DP

    @property
    def AllStop(self):
        """Whether all target misfit values have been met.

        Returns
        -------
        bool
        """

        return self.DM and self.CL and self.DP

    @property
    def DMtarget(self):
        if getattr(self, "_DMtarget", None) is None:
            self._DMtarget = self.chifact * self.phi_d_star
        return self._DMtarget

    @DMtarget.setter
    def DMtarget(self, val):
        self._DMtarget = val

    @property
    def CLtarget(self):
        if not getattr(self.pgi_smallness, "approx_eval", True):
            # if nonlinear prior, compute targer numerically at each GMM update
            samples, _ = self.pgi_smallness.gmm.sample(
                len(self.pgi_smallness.gmm.cell_volumes)
            )
            self.phi_ms_star = self.pgi_smallness(
                mkvc(samples), externalW=self.WeightsInTarget
            )

            self._CLtarget = self.chiSmall * self.phi_ms_star

        elif getattr(self, "_CLtarget", None) is None:
            # the factor of 0.5 is because we do phid = 0.5*|| dpred - dobs||^2
            if self.phi_ms_star is None:
                # Expected value is number of active cells * number of physical
                # properties
                self.phi_ms_star = 0.5 * len(self.invProb.model)

            self._CLtarget = self.chiSmall * self.phi_ms_star

        return self._CLtarget

    @property
    def CLnormalizedConstant(self):
        if ~self.WeightsInTarget:
            return 1.0
        elif np.any(self.smallness == -1):
            return np.sum(
                sp.csr_matrix.diagonal(self.invProb.reg.objfcts[0].W) ** 2.0
            ) / len(self.invProb.model)
        else:
            return np.sum(sp.csr_matrix.diagonal(self.pgi_smallness.W) ** 2.0) / len(
                self.invProb.model
            )

    @CLtarget.setter
    def CLtarget(self, val):
        self._CLtarget = val

    def phims(self):
        if np.any(self.smallness == -1):
            return self.invProb.reg.objfcts[0](self.invProb.model)
        else:
            return (
                self.pgi_smallness(
                    self.invProb.model, external_weights=self.WeightsInTarget
                )
                / self.CLnormalizedConstant
            )

    def ThetaTarget(self):
        maxdiff = 0.0

        for i in range(self.invProb.reg.gmm.n_components):
            meandiff = np.linalg.norm(
                (self.invProb.reg.gmm.means_[i] - self.invProb.reg.gmmref.means_[i])
                / self.invProb.reg.gmmref.means_[i],
                ord=self.distance_norm,
            )
            maxdiff = np.maximum(maxdiff, meandiff)

            if (
                self.invProb.reg.gmm.covariance_type == "full"
                or self.invProb.reg.gmm.covariance_type == "spherical"
            ):
                covdiff = np.linalg.norm(
                    (
                        self.invProb.reg.gmm.covariances_[i]
                        - self.invProb.reg.gmmref.covariances_[i]
                    )
                    / self.invProb.reg.gmmref.covariances_[i],
                    ord=self.distance_norm,
                )
            else:
                covdiff = np.linalg.norm(
                    (
                        self.invProb.reg.gmm.covariances_
                        - self.invProb.reg.gmmref.covariances_
                    )
                    / self.invProb.reg.gmmref.covariances_,
                    ord=self.distance_norm,
                )
            maxdiff = np.maximum(maxdiff, covdiff)

            pidiff = np.linalg.norm(
                [
                    (
                        self.invProb.reg.gmm.weights_[i]
                        - self.invProb.reg.gmmref.weights_[i]
                    )
                    / self.invProb.reg.gmmref.weights_[i]
                ],
                ord=self.distance_norm,
            )
            maxdiff = np.maximum(maxdiff, pidiff)

        return maxdiff

    def endIter(self):

        self._DM = False
        self._CL = True
        self._DP = True
        self.dmlist = np.r_[[dmis(self.invProb.model) for dmis in self.dmisfit.objfcts]]
        self.targetlist = np.r_[
            [dm < tgt for dm, tgt in zip(self.dmlist, self.DMtarget)]
        ]

        if np.all(self.targetlist):
            self._DM = True

        if self.TriggerSmall and np.any(self.smallness != -1):
            if self.phims() > self.CLtarget:
                self._CL = False

        if self.TriggerTheta:
            if self.ThetaTarget() > self.ToleranceTheta:
                self._DP = False

        if self.verbose:
            message = "geophys. misfits: " + "; ".join(
                map(
                    str,
                    [
                        "{0} (target {1} [{2}])".format(val, tgt, cond)
                        for val, tgt, cond in zip(
                            np.round(self.dmlist, 1),
                            np.round(self.DMtarget, 1),
                            self.targetlist,
                        )
                    ],
                )
            )
            if self.TriggerSmall:
                message += (
                    " | smallness misfit: {0:.1f} (target: {1:.1f} [{2}])".format(
                        self.phims(), self.CLtarget, self.CL
                    )
                )
            if self.TriggerTheta:
                message += " | GMM parameters within tolerance: {}".format(self.DP)
            print(message)

        if self.AllStop:
            self.opt.stopNextIteration = True
            if self.verbose:
                print("All targets have been reached")


class SaveEveryIteration(InversionDirective):
    """SaveEveryIteration

    This directive saves an array at each iteration. The default
    directory is the current directory and the models are saved as
    ``InversionModel-YYYY-MM-DD-HH-MM-iter.npy``
    """

    def __init__(self, directory=".", name="InversionModel", **kwargs):
        super().__init__(**kwargs)
        self.directory = directory
        self.name = name

    @property
    def directory(self):
        """Directory to save results in.

        Returns
        -------
        str
        """
        return self._directory

    @directory.setter
    def directory(self, value):
        value = validate_string("directory", value)
        fullpath = os.path.abspath(os.path.expanduser(value))

        if not os.path.isdir(fullpath):
            os.mkdir(fullpath)
        self._directory = value

    @property
    def name(self):
        """Root of the filename to be saved.

        Returns
        -------
        str
        """
        return self._name

    @name.setter
    def name(self, value):
        self._name = validate_string("name", value)

    @property
    def fileName(self):
        if getattr(self, "_fileName", None) is None:
            from datetime import datetime

            self._fileName = "{0!s}-{1!s}".format(
                self.name, datetime.now().strftime("%Y-%m-%d-%H-%M")
            )
        return self._fileName


class SaveModelEveryIteration(SaveEveryIteration):
    """SaveModelEveryIteration

    This directive saves the model as a numpy array at each iteration. The
    default directory is the current directoy and the models are saved as
    ``InversionModel-YYYY-MM-DD-HH-MM-iter.npy``
    """

    def initialize(self):
        print(
            "SimPEG.SaveModelEveryIteration will save your models as: "
            "'{0!s}###-{1!s}.npy'".format(self.directory + os.path.sep, self.fileName)
        )

    def endIter(self):
        np.save(
            "{0!s}{1:03d}-{2!s}".format(
                self.directory + os.path.sep, self.opt.iter, self.fileName
            ),
            self.opt.xc,
        )


class SaveOutputEveryIteration(SaveEveryIteration):
    """SaveOutputEveryIteration"""

    save_txt = True

    def __init__(self, save_txt=True, **kwargs):
        super().__init__(**kwargs)

        self.save_txt = save_txt

    @property
    def save_txt(self):
        """Whether to save the output as a text file.

        Returns
        -------
        bool
        """
        return self._save_txt

    @save_txt.setter
    def save_txt(self, value):
        self._save_txt = validate_type("save_txt", value, bool)

    def initialize(self):
        if self.save_txt is True:
            print(
                "SimPEG.SaveOutputEveryIteration will save your inversion "
                "progress as: '###-{0!s}.txt'".format(self.fileName)
            )
            f = open(self.fileName + ".txt", "w")
            header = "  #     beta     phi_d     phi_m   phi_m_small     phi_m_smoomth_x     phi_m_smoomth_y     phi_m_smoomth_z      phi\n"
            f.write(header)
            f.close()

        # Create a list of each

        self.beta = []
        self.phi_d = []
        self.phi_m = []
        self.phi_m_small = []
        self.phi_m_smooth_x = []
        self.phi_m_smooth_y = []
        self.phi_m_smooth_z = []
        self.phi = []

    def endIter(self):

        phi_s, phi_x, phi_y, phi_z = 0, 0, 0, 0

        for reg in self.reg.objfcts:
            if isinstance(reg, Sparse):
                i_s, i_x, i_y, i_z = 0, 1, 2, 3
            else:
                i_s, i_x, i_y, i_z = 0, 1, 3, 5
            if getattr(reg, "alpha_s", None):
                phi_s += reg.objfcts[i_s](self.invProb.model) * reg.alpha_s
            if getattr(reg, "alpha_x", None):
                phi_x += reg.objfcts[i_x](self.invProb.model) * reg.alpha_x

            if reg.regularization_mesh.dim > 1 and getattr(reg, "alpha_y", None):
                phi_y += reg.objfcts[i_y](self.invProb.model) * reg.alpha_y
            if reg.regularization_mesh.dim > 2 and getattr(reg, "alpha_z", None):
                phi_z += reg.objfcts[i_z](self.invProb.model) * reg.alpha_z

        self.beta.append(self.invProb.beta)
        self.phi_d.append(self.invProb.phi_d)
        self.phi_m.append(self.invProb.phi_m)
        self.phi_m_small.append(phi_s)
        self.phi_m_smooth_x.append(phi_x)
        self.phi_m_smooth_y.append(phi_y)
        self.phi_m_smooth_z.append(phi_z)
        self.phi.append(self.opt.f)

        if self.save_txt:
            f = open(self.fileName + ".txt", "a")
            f.write(
                " {0:3d} {1:1.4e} {2:1.4e} {3:1.4e} {4:1.4e} {5:1.4e} "
                "{6:1.4e}  {7:1.4e}  {8:1.4e}\n".format(
                    self.opt.iter,
                    self.beta[self.opt.iter - 1],
                    self.phi_d[self.opt.iter - 1],
                    self.phi_m[self.opt.iter - 1],
                    self.phi_m_small[self.opt.iter - 1],
                    self.phi_m_smooth_x[self.opt.iter - 1],
                    self.phi_m_smooth_y[self.opt.iter - 1],
                    self.phi_m_smooth_z[self.opt.iter - 1],
                    self.phi[self.opt.iter - 1],
                )
            )
            f.close()

    def load_results(self):
        results = np.loadtxt(self.fileName + str(".txt"), comments="#")
        self.beta = results[:, 1]
        self.phi_d = results[:, 2]
        self.phi_m = results[:, 3]
        self.phi_m_small = results[:, 4]
        self.phi_m_smooth_x = results[:, 5]
        self.phi_m_smooth_y = results[:, 6]
        self.phi_m_smooth_z = results[:, 7]

        self.phi_m_smooth = (
            self.phi_m_smooth_x + self.phi_m_smooth_y + self.phi_m_smooth_z
        )

        self.f = results[:, 7]

        self.target_misfit = self.invProb.dmisfit.simulation.survey.nD / 2.0
        self.i_target = None

        if self.invProb.phi_d < self.target_misfit:
            i_target = 0
            while self.phi_d[i_target] > self.target_misfit:
                i_target += 1
            self.i_target = i_target

    def plot_misfit_curves(
        self,
        fname=None,
        dpi=300,
        plot_small_smooth=False,
        plot_phi_m=True,
        plot_small=False,
        plot_smooth=False,
    ):

        self.target_misfit = self.invProb.dmisfit.simulation.survey.nD / 2.0
        self.i_target = None

        if self.invProb.phi_d < self.target_misfit:
            i_target = 0
            while self.phi_d[i_target] > self.target_misfit:
                i_target += 1
            self.i_target = i_target

        fig = plt.figure(figsize=(5, 2))
        ax = plt.subplot(111)
        ax_1 = ax.twinx()
        ax.semilogy(
            np.arange(len(self.phi_d)), self.phi_d, "k-", lw=2, label=r"$\phi_d$"
        )

        if plot_phi_m:
            ax_1.semilogy(
                np.arange(len(self.phi_d)), self.phi_m, "r", lw=2, label=r"$\phi_m$"
            )

        if plot_small_smooth or plot_small:
            ax_1.semilogy(
                np.arange(len(self.phi_d)), self.phi_m_small, "ro", label="small"
            )
        if plot_small_smooth or plot_smooth:
            ax_1.semilogy(
                np.arange(len(self.phi_d)), self.phi_m_smooth_x, "rx", label="smooth_x"
            )
            ax_1.semilogy(
                np.arange(len(self.phi_d)), self.phi_m_smooth_y, "rx", label="smooth_y"
            )
            ax_1.semilogy(
                np.arange(len(self.phi_d)), self.phi_m_smooth_z, "rx", label="smooth_z"
            )

        ax.legend(loc=1)
        ax_1.legend(loc=2)

        ax.plot(
            np.r_[ax.get_xlim()[0], ax.get_xlim()[1]],
            np.ones(2) * self.target_misfit,
            "k:",
        )
        ax.set_xlabel("Iteration")
        ax.set_ylabel(r"$\phi_d$")
        ax_1.set_ylabel(r"$\phi_m$", color="r")
        ax_1.tick_params(axis="y", which="both", colors="red")

        plt.show()
        if fname is not None:
            fig.savefig(fname, dpi=dpi)

    def plot_tikhonov_curves(self, fname=None, dpi=200):

        self.target_misfit = self.invProb.dmisfit.simulation.survey.nD / 2.0
        self.i_target = None

        if self.invProb.phi_d < self.target_misfit:
            i_target = 0
            while self.phi_d[i_target] > self.target_misfit:
                i_target += 1
            self.i_target = i_target

        fig = plt.figure(figsize=(5, 8))
        ax1 = plt.subplot(311)
        ax2 = plt.subplot(312)
        ax3 = plt.subplot(313)

        ax1.plot(self.beta, self.phi_d, "k-", lw=2, ms=4)
        ax1.set_xlim(np.hstack(self.beta).min(), np.hstack(self.beta).max())
        ax1.set_xlabel(r"$\beta$", fontsize=14)
        ax1.set_ylabel(r"$\phi_d$", fontsize=14)

        ax2.plot(self.beta, self.phi_m, "k-", lw=2)
        ax2.set_xlim(np.hstack(self.beta).min(), np.hstack(self.beta).max())
        ax2.set_xlabel(r"$\beta$", fontsize=14)
        ax2.set_ylabel(r"$\phi_m$", fontsize=14)

        ax3.plot(self.phi_m, self.phi_d, "k-", lw=2)
        ax3.set_xlim(np.hstack(self.phi_m).min(), np.hstack(self.phi_m).max())
        ax3.set_xlabel(r"$\phi_m$", fontsize=14)
        ax3.set_ylabel(r"$\phi_d$", fontsize=14)

        if self.i_target is not None:
            ax1.plot(self.beta[self.i_target], self.phi_d[self.i_target], "k*", ms=10)
            ax2.plot(self.beta[self.i_target], self.phi_m[self.i_target], "k*", ms=10)
            ax3.plot(self.phi_m[self.i_target], self.phi_d[self.i_target], "k*", ms=10)

        for ax in [ax1, ax2, ax3]:
            ax.set_xscale("linear")
            ax.set_yscale("linear")
        plt.tight_layout()
        plt.show()
        if fname is not None:
            fig.savefig(fname, dpi=dpi)


class SaveOutputDictEveryIteration(SaveEveryIteration):
    """
    Saves inversion parameters at every iteration.
    """

    # Initialize the output dict
    def __init__(self, saveOnDisk=False, **kwargs):
        super().__init__(**kwargs)
        self.saveOnDisk = saveOnDisk

    @property
    def saveOnDisk(self):
        """Whether to save the output dict to disk.

        Returns
        -------
        bool
        """
        return self._saveOnDisk

    @saveOnDisk.setter
    def saveOnDisk(self, value):
        self._saveOnDisk = validate_type("saveOnDisk", value, bool)

    def initialize(self):
        self.outDict = {}
        if self.saveOnDisk:
            print(
                "SimPEG.SaveOutputDictEveryIteration will save your inversion progress as dictionary: '###-{0!s}.npz'".format(
                    self.fileName
                )
            )

    def endIter(self):

        # regCombo = ["phi_ms", "phi_msx"]

        # if self.simulation[0].mesh.dim >= 2:
        #     regCombo += ["phi_msy"]

        # if self.simulation[0].mesh.dim == 3:
        #     regCombo += ["phi_msz"]

        # Initialize the output dict
        iterDict = {}

        # Save the data.
        iterDict["iter"] = self.opt.iter
        iterDict["beta"] = self.invProb.beta
        iterDict["phi_d"] = self.invProb.phi_d
        iterDict["phi_m"] = self.invProb.phi_m

        # for label, fcts in zip(regCombo, self.reg.objfcts[0].objfcts):
        #     iterDict[label] = fcts(self.invProb.model)

        iterDict["f"] = self.opt.f
        iterDict["m"] = self.invProb.model
        iterDict["dpred"] = self.invProb.dpred

        if hasattr(self.reg.objfcts[0], "eps_p") is True:
            iterDict["eps_p"] = self.reg.objfcts[0].eps_p
            iterDict["eps_q"] = self.reg.objfcts[0].eps_q

        if hasattr(self.reg.objfcts[0], "norms") is True:
            iterDict["lps"] = self.reg.objfcts[0].norms[0][0]
            iterDict["lpx"] = self.reg.objfcts[0].norms[0][1]

        # Save the file as a npz
        if self.saveOnDisk:

            np.savez("{:03d}-{:s}".format(self.opt.iter, self.fileName), iterDict)

        self.outDict[self.opt.iter] = iterDict


class Update_IRLS(InversionDirective):

    f_old = 0
    f_min_change = 1e-2
    beta_tol = 1e-1
    beta_ratio_l2 = None
    prctile = 100
    chifact_start = 1.0
    chifact_target = 1.0

    # Solving parameter for IRLS (mode:2)
    irls_iteration = 0
    minGNiter = 1
    iterStart = 0
    sphericalDomain = False

    # Beta schedule
    ComboObjFun = False
    mode = 1
    coolEpsOptimized = True
    coolEps_p = True
    coolEps_q = True
    floorEps_p = 1e-8
    floorEps_q = 1e-8
    coolEpsFact = 1.2
    silent = False
    fix_Jmatrix = False

    def __init__(
        self,
        max_irls_iterations=20,
        update_beta=True,
        beta_search=False,
        coolingFactor=2.0,
        coolingRate=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_irls_iterations = max_irls_iterations
        self.update_beta = update_beta
        self.beta_search = beta_search
        self.coolingFactor = coolingFactor
        self.coolingRate = coolingRate

    @property
    def max_irls_iterations(self):
        """Maximum irls iterations.

        Returns
        -------
        int
        """
        return self._max_irls_iterations

    @max_irls_iterations.setter
    def max_irls_iterations(self, value):
        self._max_irls_iterations = validate_integer(
            "max_irls_iterations", value, min_val=0
        )

    @property
    def coolingFactor(self):
        """Beta is divided by this value every `coolingRate` iterations.

        Returns
        -------
        float
        """
        return self._coolingFactor

    @coolingFactor.setter
    def coolingFactor(self, value):
        self._coolingFactor = validate_float(
            "coolingFactor", value, min_val=0.0, inclusive_min=False
        )

    @property
    def coolingRate(self):
        """Cool after this number of iterations.

        Returns
        -------
        int
        """
        return self._coolingRate

    @coolingRate.setter
    def coolingRate(self, value):
        self._coolingRate = validate_integer("coolingRate", value, min_val=1)

    @property
    def update_beta(self):
        """Whether to update beta.

        Returns
        -------
        bool
        """
        return self._update_beta

    @update_beta.setter
    def update_beta(self, value):
        self._update_beta = validate_type("update_beta", value, bool)

    @property
    def beta_search(self):
        """Whether to do a beta search.

        Returns
        -------
        bool
        """
        return self._beta_search

    @beta_search.setter
    def beta_search(self, value):
        self._beta_search = validate_type("beta_search", value, bool)

    @property
    def target(self):
        if getattr(self, "_target", None) is None:
            nD = 0
            for survey in self.survey:
                nD += survey.nD

            self._target = nD * 0.5 * self.chifact_target

        return self._target

    @target.setter
    def target(self, val):
        self._target = val

    @property
    def start(self):
        if getattr(self, "_start", None) is None:
            if isinstance(self.survey, list):
                self._start = 0
                for survey in self.survey:
                    self._start += survey.nD * 0.5 * self.chifact_start

            else:

                self._start = self.survey.nD * 0.5 * self.chifact_start
        return self._start

    @start.setter
    def start(self, val):
        self._start = val

    def initialize(self):

        if self.mode == 1:

            self.norms = []
            for reg in self.reg.objfcts:
                self.norms.append(reg.norms)
                reg.norms = [2.0 for obj in reg.objfcts]
                reg.model = self.invProb.model

        # Update the model used by the regularization
        for reg in self.reg.objfcts:
            reg.model = self.invProb.model

        if self.sphericalDomain:
            self.angleScale()

    def endIter(self):

        if self.sphericalDomain:
            self.angleScale()

        # Check if misfit is within the tolerance, otherwise scale beta
        if np.all(
            [
                np.abs(1.0 - self.invProb.phi_d / self.target) > self.beta_tol,
                self.update_beta,
                self.mode != 1,
            ]
        ):
            ratio = self.target / self.invProb.phi_d

            if ratio > 1:
                ratio = np.mean([2.0, ratio])
            else:
                ratio = np.mean([0.75, ratio])

            self.invProb.beta = self.invProb.beta * ratio

            if np.all([self.mode != 1, self.beta_search]):
                print("Beta search step")
                # self.update_beta = False
                # Re-use previous model and continue with new beta
                self.invProb.model = self.reg.objfcts[0].model
                self.opt.xc = self.reg.objfcts[0].model
                self.opt.iter -= 1
                return

        elif np.all([self.mode == 1, self.opt.iter % self.coolingRate == 0]):
            self.invProb.beta = self.invProb.beta / self.coolingFactor

        # After reaching target misfit with l2-norm, switch to IRLS (mode:2)
        if np.all([self.invProb.phi_d < self.start, self.mode == 1]):
            self.start_irls()

        # Only update after GN iterations
        if np.all(
            [(self.opt.iter - self.iterStart) % self.minGNiter == 0, self.mode != 1]
        ):
            if self.stopping_criteria():
                self.opt.stopNextIteration = True
                return

            # Print to screen
            for reg in self.reg.objfcts:
                for obj in reg.objfcts:
                    if isinstance(reg, (Sparse, BaseSparse)):
                        obj.irls_threshold = obj.irls_threshold / self.coolEpsFact

            self.irls_iteration += 1

            # Reset the regularization matrices so that it is
            # recalculated for current model. Do it to all levels of comboObj
            for reg in self.reg.objfcts:
                if isinstance(reg, (Sparse, BaseSparse)):
                    reg.update_weights(reg.model)

            self.update_beta = True
            self.invProb.phi_m_last = self.reg(self.invProb.model)

    def start_irls(self):
        if not self.silent:
            print(
                "Reached starting chifact with l2-norm regularization:"
                + " Start IRLS steps..."
            )

        self.mode = 2

        if getattr(self.opt, "iter", None) is None:
            self.iterStart = 0
        else:
            self.iterStart = self.opt.iter

        self.invProb.phi_m_last = self.reg(self.invProb.model)

        # Either use the supplied irls_threshold, or fix base on distribution of
        # model values
        for reg in self.reg.objfcts:

            if not isinstance(reg, Sparse):
                continue

            for obj in reg.objfcts:

                threshold = np.percentile(
                    np.abs(obj.mapping * obj._delta_m(self.invProb.model)), self.prctile
                )

                if isinstance(obj, SmoothnessFirstOrder):
                    threshold /= reg.regularization_mesh.base_length

                obj.irls_threshold = threshold

        # Re-assign the norms supplied by user l2 -> lp
        for reg, norms in zip(self.reg.objfcts, self.norms):
            reg.norms = norms

        # Save l2-model
        self.invProb.l2model = self.invProb.model.copy()

        # Print to screen
        for reg in self.reg.objfcts:
            if not self.silent:
                print("irls_threshold " + str(reg.objfcts[0].irls_threshold))

    def angleScale(self):
        """
        Update the scales used by regularization for the
        different block of models
        """
        # Currently implemented for MVI-S only
        max_p = []
        for reg in self.reg.objfcts[0].objfcts:
            f_m = abs(reg.f_m(reg.model))
            max_p += [np.max(f_m)]

        max_p = np.asarray(max_p).max()

        max_s = [np.pi, np.pi]

        for reg, var in zip(self.reg.objfcts[1:], max_s):
            for obj in reg.objfcts:
                # TODO Need to make weights_shapes a public method
                obj.set_weights(
                    angle_scale=np.ones(obj._weights_shapes[0]) * max_p / var
                )

    def validate(self, directiveList):
        # check if a linear preconditioner is in the list, if not warn else
        # assert that it is listed after the IRLS directive
        dList = directiveList.dList
        self_ind = dList.index(self)
        lin_precond_ind = [isinstance(d, UpdatePreconditioner) for d in dList]

        if any(lin_precond_ind):
            assert lin_precond_ind.index(True) > self_ind, (
                "The directive 'UpdatePreconditioner' must be after Update_IRLS "
                "in the directiveList"
            )
        else:
            warnings.warn(
                "Without a Linear preconditioner, convergence may be slow. "
                "Consider adding `Directives.UpdatePreconditioner` to your "
                "directives list"
            )
        return True

    def stopping_criteria(self):
        """
        Check for stopping criteria of max_irls_iteration or minimum change.
        """
        phim_new = 0
        for reg in self.reg.objfcts:
            if isinstance(reg, (Sparse, BaseSparse)):
                reg.model = self.invProb.model
                phim_new += reg(reg.model)

        # Check for maximum number of IRLS cycles1
        if self.irls_iteration == self.max_irls_iterations:
            if not self.silent:
                print(
                    "Reach maximum number of IRLS cycles:"
                    + " {0:d}".format(self.max_irls_iterations)
                )
            return True

        # Check if the function has changed enough
        f_change = np.abs(self.f_old - phim_new) / (self.f_old + 1e-12)
        if np.all(
            [
                f_change < self.f_min_change,
                self.irls_iteration > 1,
                np.abs(1.0 - self.invProb.phi_d / self.target) < self.beta_tol,
            ]
        ):
            print("Minimum decrease in regularization." + "End of IRLS")
            return True

        self.f_old = phim_new

        return False


class UpdatePreconditioner(InversionDirective):
    """
    Create a Jacobi preconditioner for the linear problem
    """

    def __init__(self, update_every_iteration=True, **kwargs):
        super().__init__(**kwargs)
        self.update_every_iteration = update_every_iteration

    @property
    def update_every_iteration(self):
        """Whether to update the preconditioner at every iteration.

        Returns
        -------
        bool
        """
        return self._update_every_iteration

    @update_every_iteration.setter
    def update_every_iteration(self, value):
        self._update_every_iteration = validate_type(
            "update_every_iteration", value, bool
        )

    def initialize(self):

        # Create the pre-conditioner
        regDiag = np.zeros_like(self.invProb.model)
        m = self.invProb.model

        for reg in self.reg.objfcts:
            # Check if regularization has a projection
            rdg = reg.deriv2(m)
            if not isinstance(rdg, Zero):
                regDiag += rdg.diagonal()

        JtJdiag = np.zeros_like(self.invProb.model)
        for sim, dmisfit in zip(self.simulation, self.dmisfit.objfcts):

            if getattr(sim, "getJtJdiag", None) is None:
                assert getattr(sim, "getJ", None) is not None, (
                    "Simulation does not have a getJ attribute."
                    + "Cannot form the sensitivity explicitly"
                )
                JtJdiag += np.sum(np.power((dmisfit.W * sim.getJ(m)), 2), axis=0)
            else:
                JtJdiag += sim.getJtJdiag(m, W=dmisfit.W)

        diagA = JtJdiag + self.invProb.beta * regDiag
        diagA[diagA != 0] = diagA[diagA != 0] ** -1.0
        PC = sdiag((diagA))

        self.opt.approxHinv = PC

    def endIter(self):
        # Cool the threshold parameter
        if self.update_every_iteration is False:
            return

        # Create the pre-conditioner
        regDiag = np.zeros_like(self.invProb.model)
        m = self.invProb.model

        for reg in self.reg.objfcts:
            # Check if he has wire
            regDiag += reg.deriv2(m).diagonal()

        JtJdiag = np.zeros_like(self.invProb.model)
        for sim, dmisfit in zip(self.simulation, self.dmisfit.objfcts):

            if getattr(sim, "getJtJdiag", None) is None:
                assert getattr(sim, "getJ", None) is not None, (
                    "Simulation does not have a getJ attribute."
                    + "Cannot form the sensitivity explicitly"
                )
                JtJdiag += np.sum(np.power((dmisfit.W * sim.getJ(m)), 2), axis=0)
            else:
                JtJdiag += sim.getJtJdiag(m, W=dmisfit.W)

        diagA = JtJdiag + self.invProb.beta * regDiag
        diagA[diagA != 0] = diagA[diagA != 0] ** -1.0
        PC = sdiag((diagA))
        self.opt.approxHinv = PC


class Update_Wj(InversionDirective):
    """
    Create approx-sensitivity base weighting using the probing method
    """

    def __init__(self, k=None, itr=None, **kwargs):
        self.k = k
        self.itr = itr
        super().__init__(**kwargs)

    @property
    def k(self):
        """Number of probing cycles for the estimator.

        Returns
        -------
        int
        """
        return self._k

    @k.setter
    def k(self, value):
        if value is not None:
            value = validate_integer("k", value, min_val=1)
        self._k = value

    @property
    def itr(self):
        """Which iteration to update the sensitivity.

        Will always update if `None`.

        Returns
        -------
        int or None
        """
        return self._itr

    @itr.setter
    def itr(self, value):
        if value is not None:
            value = validate_integer("itr", value, min_val=1)
        self._itr = value

    def endIter(self):

        if self.itr is None or self.itr == self.opt.iter:

            m = self.invProb.model
            if self.k is None:
                self.k = int(self.survey.nD / 10)

            def JtJv(v):

                Jv = self.simulation.Jvec(m, v)

                return self.simulation.Jtvec(m, Jv)

            JtJdiag = diagEst(JtJv, len(m), k=self.k)
            JtJdiag = JtJdiag / max(JtJdiag)

            self.reg.wght = JtJdiag


class UpdateSensitivityWeights(InversionDirective):
    """
    Directive to take care of re-weighting
    the non-linear problems. Assumes that the map of the regularization
    function is either Wires or Identity.
    Good for any problem where J is formed explicitly.
    """

    def __init__(
            self,
            everyIter=True,
            threshold=1e-12,
            normalization=True,
            method="percent_amplitude",
            **kwargs
    ):
        super().__init__(**kwargs)
        self.everyIter = everyIter
        self.threshold = threshold
        self.method = method
        self.normalization = normalization

    @property
    def everyIter(self):
        """Whether to update the sensitivity weights at every iteration.

        Returns
        -------
        bool
        """
        return self._everyIter

    @everyIter.setter
    def everyIter(self, value):
        self._everyIter = validate_type("everyIter", value, bool)

    @property
    def threshold(self):
        """A small threshold added to the weights to represent a minimum weighting value.

        Returns
        -------
        float
        """
        return self._threshold

    @threshold.setter
    def threshold(self, value):
        self._threshold = validate_float("threshold", value, min_val=0.0)

    @property
    def normalization(self):
        """Whether to normalize by the smallest sensitivity weight.

        Returns
        -------
        bool
        """
        return self._normalization

    @normalization.setter
    def normalization(self, value):
        self._normalization = validate_type("normalization", value, bool)

    def initialize(self):
        """
        Calculate and update sensitivity
        for optimization and regularization
        """
        for reg in self.reg.objfcts:
            if not isinstance(reg.mapping, (IdentityMap, Wires)):
                raise TypeError(
                    f"Mapping for the regularization must be of type {IdentityMap} or {Wires}. "
                    + f"Input mapping of type {type(reg.mapping)}."
                )

        self.update()

    def endIter(self):
        """
        Update inverse problem
        """
        if self.everyIter:
            self.update()

    def update(self):
        """
        Compute explicitly the main diagonal of JtJ

        """
        jtj_diag = np.zeros_like(self.invProb.model)
        m = self.invProb.model

        for sim, dmisfit in zip(self.simulation, self.dmisfit.objfcts):
            if getattr(sim, "getJtJdiag", None) is None:
                if getattr(sim, "getJ", None) is None:
                    raise AttributeError(
                        "Simulation does not have a getJ attribute."
                        + "Cannot form the sensitivity explicitly"
                    )
                jtj_diag += mkvc(np.sum((dmisfit.W * sim.getJ(m)) ** 2.0, axis=0))
            else:
                jtj_diag += sim.getJtJdiag(m, W=dmisfit.W)

        # Normalize and threshold weights
        wr = np.zeros_like(self.invProb.model)
        for reg in self.reg.objfcts:
            if not isinstance(reg, BaseSimilarityMeasure):
                wr += reg.mapping.deriv(self.invProb.model).T * (
                    (reg.mapping * jtj_diag) / reg.regularization_mesh.vol ** 2.0
                )
        if self.normalization:
            wr /= wr.max()

        wr **= 0.5
        if self.method == "percent_amplitude":
            wr += np.abs(wr.max() - wr.min()) * self.threshold / 100.  # #np.percentile(wr, self.threshold) #threshold
        else:
            wr += self.threshold

        for reg in self.reg.objfcts:
            if not isinstance(reg, BaseSimilarityMeasure):
                sub_regs = getattr(reg, "objfcts", [reg])
                for sub_reg in sub_regs:
                    sub_reg.set_weights(sensitivity=sub_reg.mapping * wr)

    def validate(self, directiveList):
        # check if a beta estimator is in the list after setting the weights
        dList = directiveList.dList
        self_ind = dList.index(self)

        beta_estimator_ind = [isinstance(d, BetaEstimate_ByEig) for d in dList]

        lin_precond_ind = [isinstance(d, UpdatePreconditioner) for d in dList]

        if any(beta_estimator_ind):
            assert beta_estimator_ind.index(True) > self_ind, (
                "The directive 'BetaEstimate_ByEig' must be after UpdateSensitivityWeights "
                "in the directiveList"
            )

        if any(lin_precond_ind):
            assert lin_precond_ind.index(True) > self_ind, (
                "The directive 'UpdatePreconditioner' must be after UpdateSensitivityWeights "
                "in the directiveList"
            )

        return True


class ProjectSphericalBounds(InversionDirective):
    r"""
    Trick for spherical coordinate system.
    Project :math:`\theta` and :math:`\phi` angles back to :math:`[-\pi,\pi]`
    using back and forth conversion.
    spherical->cartesian->spherical
    """

    def initialize(self):

        x = self.invProb.model
        # Convert to cartesian than back to avoid over rotation
        nC = int(len(x) / 3)

        xyz = spherical2cartesian(x.reshape((nC, 3), order="F"))
        m = cartesian2spherical(xyz.reshape((nC, 3), order="F"))

        self.invProb.model = m

        for sim in self.simulation:
            sim.model = m

        self.opt.xc = m

    def endIter(self):

        x = self.invProb.model
        nC = int(len(x) / 3)

        # Convert to cartesian than back to avoid over rotation
        xyz = spherical2cartesian(x.reshape((nC, 3), order="F"))
        m = cartesian2spherical(xyz.reshape((nC, 3), order="F"))

        self.invProb.model = m

        phi_m_last = []
        for reg in self.reg.objfcts:
            reg.model = self.invProb.model
            phi_m_last += [reg(self.invProb.model)]

        self.invProb.phi_m_last = phi_m_last

        for sim in self.simulation:
            sim.model = m

        self.opt.xc = m
