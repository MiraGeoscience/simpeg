import numpy as np
import scipy.sparse as sp

from .base import BaseSimilarityMeasure
from ..utils import validate_type
from ..utils.mat_utils import coterminal

###############################################################################
#                                                                             #
#                                Cross-Gradient                               #
#                                                                             #
###############################################################################


class CrossGradient(BaseSimilarityMeasure):
    r"""
    The cross-gradient constraint for joint inversions.

    ..math::
        \phi_c(\mathbf{m_1},\mathbf{m_2}) = \lambda \sum_{i=1}^{M} \|
        \nabla \mathbf{m_1}_i \times \nabla \mathbf{m_2}_i \|^2

    All methods assume that we are working with two models only.

    """

    def __init__(self, mesh, wire_map, approx_hessian=True, normalize=False, **kwargs):
        super().__init__(mesh, wire_map=wire_map, **kwargs)
        self.approx_hessian = approx_hessian
        self._units = ["metric", "metric"]
        self.normalize = normalize
        regmesh = self.regularization_mesh
        self.set_weights(volume=self.regularization_mesh.vol)

        if regmesh.mesh.dim not in (2, 3):
            raise ValueError("Cross-Gradient is only defined for 2D or 3D")
        self._G = regmesh.cell_gradient
        self._Av = regmesh.average_face_to_cell

    @property
    def approx_hessian(self):
        """whether to use the semi-positive definate approximation for the hessian.
        Returns
        -------
        bool
        """
        return self._approx_hessian

    @approx_hessian.setter
    def approx_hessian(self, value):
        self._approx_hessian = validate_type("approx_hessian", value, bool)

    def _model_gradients(self, models):
        """
        Compute gradient on faces
        """
        gradients = []
        for unit, wire in zip(self.units, self.wire_map):
            model = wire * models
            if unit == "radian":
                gradient = []
                components = "xyz" if self.regularization_mesh.dim == 3 else "xy"
                for comp in components:
                    distances = getattr(
                        self.regularization_mesh, f"cell_distances_{comp}"
                    )
                    cell_grad = getattr(
                        self.regularization_mesh, f"cell_gradient_{comp}"
                    )
                    gradient.append(
                        coterminal(cell_grad * model * distances) / distances
                    )

                gradient = np.hstack(gradient) / np.pi
            else:
                gradient = self._G @ model

            gradients.append(gradient)

        return gradients

    def _calculate_gradient(self, model, normalized=False, rtol=1e-6):
        """
        Calculate the spatial gradients of the model using central difference.

        Concatenates gradient components into a single array.
        [[x_grad1, y_grad1, z_grad1],
         [x_grad2, y_grad2, z_grad2],
         [x_grad3, y_grad3, z_grad3],...]

        :param numpy.ndarray model: model

        :rtype: numpy.ndarray
        :return: gradient_vector: array where each row represents a model cell,
                 and each column represents a component of the gradient.

        """
        regmesh = self.regularization_mesh
        Avs = [regmesh.aveFx2CC, regmesh.aveFy2CC]
        if regmesh.dim == 3:
            Avs.append(regmesh.aveFz2CC)
        Av = sp.block_diag(Avs)

        # Compute the gradients and concatenate components.
        grad_models = self._model_gradients(model)

        gradients = []
        for gradient in grad_models:
            gradient = (Av @ (gradient)).reshape((-1, regmesh.dim), order="F")

            if normalized:
                norms = np.linalg.norm(gradient, axis=-1)
                ind = norms <= norms.max() * rtol
                norms[ind] = 1.0
                gradient /= norms[:, None]
                gradient[ind] = 0.0
                # set gradient to 0 if amplitude of gradient is extremely small
            gradients.append(gradient)

        return gradients

    def calculate_cross_gradient(self, model, normalized=False, rtol=1e-6):
        """
        Calculates the cross-gradients of the models at each cell center.

        Parameters
        ----------
        model : numpy.ndarray
            The input model, which will be automatically separated into the two
            parameters internally
        normalized : bool, optional
            Whether to normalize the gradient
        rtol : float, optional
            relative cuttoff for small gradients in the normalization

        Returns
        -------
        cross_grad : numpy.ndarray
            The norm of the cross gradient vector in each active cell.
        """
        # Compute the gradients and concatenate components.
        grad_m1, grad_m2 = self._calculate_gradient(
            model, normalized=normalized, rtol=rtol
        )

        # for each model cell, compute the cross product of the gradient vectors.
        cross_prod = np.cross(grad_m1, grad_m2)
        if self.regularization_mesh.dim == 3:
            cross_prod = np.linalg.norm(cross_prod, axis=-1)

        return cross_prod

    def __call__(self, model):
        r"""
        Computes the sum of all cross-gradient values at all cell centers.

        :param numpy.ndarray model: stacked array of individual models
                                    np.c_[model1, model2,...]
        :param bool normalized: returns value of normalized cross-gradient if True

        :rtype: float
        :returns: the computed value of the cross-gradient term.


        ..math::

            \phi_c(\mathbf{m_1},\mathbf{m_2})
            = \lambda \sum_{i=1}^{M} \|\nabla \mathbf{m_1}_i \times \nabla \mathbf{m_2}_i \|^2
            = \sum_{i=1}^{M} \|\nabla \mathbf{m_1}_i\|^2 \ast \|\nabla \mathbf{m_2}_i\|^2
                - (\nabla \mathbf{m_1}_i \cdot \nabla \mathbf{m_2}_i )^2
            = \|\phi_{cx}\|^2 + \|\phi_{cy}\|^2 + \|\phi_{cz}\|^2

        (optional strategy, not used in this script)

        """
        # m1, m2 = (wire * model for wire in self.wire_map)
        W1 = self.W[0]
        W2 = self.W[1]
        Av = self._Av
        G = self._G
        g_m1, g_m2 = self._model_gradients(model)

        return 0.5 * np.sum(
            (W1.T @ W1 @ Av @ g_m1**2) * (W2.T @ W2 @ Av @ g_m2**2)
            - ((W1 @ Av @ g_m1) * (W2 @ Av @ g_m2)) ** 2
        )

    def deriv(self, model):
        """
        Computes the Jacobian of the cross-gradient.

        :param list of numpy.ndarray ind_models: [model1, model2,...]

        :rtype: numpy.ndarray
        :return: result: gradient of the cross-gradient with respect to model1, model2

        """
        W1 = self.W[0]
        W2 = self.W[1]
        Av = self._Av
        G = self._G
        g_m1, g_m2 = self._model_gradients(model)

        return (
            self.wire_map_deriv.T
            * np.r_[
                (((W1 @ Av @ g_m2**2) @ W1 @ Av) * g_m1) @ G
                - (((W1 @ Av @ (g_m1 * g_m2)) @ W1 @ Av) * g_m2) @ G,
                (((W2 @ Av @ g_m1**2) @ W2 @ Av) * g_m2) @ G
                - (((W2 @ Av @ (g_m1 * g_m2)) @ W2 @ Av) * g_m1) @ G,
            ]
        )

    def deriv2(self, model, v=None):
        """
        Computes the Hessian of the cross-gradient.

        :param list of numpy.ndarray ind_models: [model1, model2, ...]
        :param numpy.ndarray v: vector to be multiplied by Hessian

        :rtype: scipy.sparse.csr_matrix if v is None
                numpy.ndarray if v is not None
        :return Hessian matrix if v is None
                Hessian multiplied by vector if v is not No

        """
        W1 = self.W[0]
        W2 = self.W[1]
        Av = self._Av
        G = self._G
        g_m1, g_m2 = self._model_gradients(model)

        if v is None:
            A = (
                G.T
                @ (
                    sp.diags(Av.T @ W1.T @ W1 @ (Av @ g_m2**2))
                    - sp.diags(g_m2) @ Av.T @ W1.T @ W1 @ Av @ sp.diags(g_m2)
                )
                @ G
            )

            C = (
                G.T
                @ (
                    sp.diags(Av.T @ W2.T @ W2 @ (Av @ g_m1**2))
                    - sp.diags(g_m1) @ Av.T @ W2.T @ W2 @ Av @ sp.diags(g_m1)
                )
                @ G
            )

            B = None
            BT = None
            if not self.approx_hessian:
                # d_m1_d_m2
                B = (
                    G.T
                    @ (
                        2 * sp.diags(g_m1) @ Av.T @ W1.T @ W1 @ Av @ sp.diags(g_m2)
                        - sp.diags(g_m2) @ Av.T @ W1.T @ W1 @ Av @ sp.diags(g_m1)
                        - sp.diags(
                            Av.T @ ((W2.T @ W2 @ Av @ g_m2) * ((W1.T @ W1 @ Av @ g_m1)))
                        )
                    )
                    @ G
                )
                BT = B.T

            return (
                self.wire_map_deriv.T
                * sp.bmat([[A, B], [BT, C]], format="csr")
                * self.wire_map_deriv
            )
        else:
            v1, v2 = (wire * v for wire in self.wire_map)

            Gv1 = G @ v1
            Gv2 = G @ v2

            p1 = G.T @ (
                (Av.T @ W1.T @ W1 @ (Av @ g_m2**2)) * Gv1
                - g_m2 * (Av.T @ W1.T @ W1 @ (Av @ (g_m2 * Gv1)))
            )
            p2 = G.T @ (
                (Av.T @ W2.T @ W2 @ (Av @ g_m1**2)) * Gv2
                - g_m1 * (Av.T @ W2.T @ W2 @ (Av @ (g_m1 * Gv2)))
            )

            if not self.approx_hessian:
                p1 += G.T @ (
                    2 * g_m1 * (Av.T @ (Av @ (g_m2 * Gv2)))
                    - g_m2 * (Av.T @ (Av @ (g_m1 * Gv2)))
                    - (Av.T @ (Av @ (g_m1 * g_m2))) * Gv2
                )

                p2 += G.T @ (
                    2 * g_m2 * (Av.T @ (Av @ (g_m1 * Gv1)))
                    - g_m1 * (Av.T @ (Av @ (g_m2 * Gv1)))
                    - (Av.T @ (Av @ (g_m2 * g_m1))) * Gv1
                )
            return self.wire_map_deriv.T * np.r_[p1, p2]

    def set_weights(self, **weights):
        """Adds (or updates) the specified weights to the regularization

        Parameters:
        -----------
        **kwargs : key, numpy.ndarray
            Each keyword argument is added to the weights used by the regularization.
            They can be accessed with their keyword argument.

        Examples
        --------
        >>> import discretize
        >>> from SimPEG.regularization import Smallness
        >>> mesh = discretize.TensorMesh([2, 3, 2])
        >>> reg = Smallness(mesh)
        >>> reg.set_weights(my_weight=np.ones(mesh.n_cells))
        >>> reg.get_weights('my_weight')
        array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
        """
        for key, values in weights.items():
            # values = validate_ndarray_with_shape(
            #     "weights", values, shape=self._weights_shapes, dtype=float
            # )
            if values.shape == self._weights_shapes:
                values = np.r_[values, values]

            self._weights[key] = values
        self._W = None

    @property
    def W(self) -> tuple:
        """
        Weighting matrix
        """
        if getattr(self, "_W", None) is None:
            weights = np.prod(list(self._weights.values()), axis=0)

            self._W = (
                sp.diags(weights[self.regularization_mesh.nC :]),
                sp.diags(weights[: self.regularization_mesh.nC]),
            )
        return self._W
