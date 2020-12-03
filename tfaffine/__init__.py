"""
Affine matrices encoded in their Lie algebra, in tensorflow.

Layers
------
AffineExp(dim: int, transform_type: str)
    .call((N, F) tensor) -> (N, dim+1, dim+1) tensor
AffineLog(dim: int, transform_type: str)
    .call((N, dim+1, dim+1) tensor) -> (N, F) tensor

Functions
---------
affine_basis(dim: int, group: str) -> (F, dim+1, dim+1) tensor
    Generate the basis of an algebra
affine_exp(prm: (N, F) tensor, basis: tensor) -> (N, dim+1, dim+1) tensor
    Exponential Lie parameters into an affine matrix
affine_log(prm: (N, dim+1, dim+1) tensor, basis: tensor) -> (N, F) tensor
    Recover Lie parameters from an affine matrix

Authors
-------
.. John Ashburner <j.ashburner@ucl.ac.uk> : original Matlab code
.. Yael Balbastre <yael.balbastre@gmail.com> : Python code

"""
import tensorflow as tf
import math


class AffineExp(tf.keras.layers.Layer):
    """Affine exponentiation layer

    Build (D+1)x(D+1) affine matrices from Lie parameters.
    """
    def __init__(self, dim, transform_type='rigid'):
        """

        Parameters
        ----------
        dim : {2, 3}
            Dimension
        transform_type : {'rigid', 'rigid+scale', 'affine'}, default='rigid'
            Transformation type
        """
        super().__init__()
        self.dim = dim
        self.transform_type = transform_type
        self.basis = None

    def build(self, input_shape):
        group = self.transform_type
        if group not in groups:
            if group == 'rigid':
                group = 'SE'
            elif group == 'rigid+scale':
                group = 'CSO'
            else:
                group = 'Aff+'
        basis = affine_basis(self.dim, group)
        self.basis = tf.Variable(initial_value=basis)

    def call(self, prm):
        """

        Parameters
        ----------
        prm : (batch, nb_param) tensor
            Parameters of the affine transform in the Lie algebra.
            `nb_param` depends on the dimension and transformation type:
            - 'rigid' : dim + (dim*(dim-1)//2)
            - 'rigid+scale' : dim + (dim*(dim-1)//2) + 1
            - 'affine' : dim * (dim + 1)
        """
        if isinstance(prm, (list, tuple)):
            prm = prm[0]
        return affine_exp(prm, self.basis)


class AffineLog(tf.keras.layers.Layer):
    """Affine logarithm layer

    Recover Lie parameters from (D+1)x(D+1) affine matrices.
    """
    def __init__(self, dim, transform_type='rigid'):
        """

        Parameters
        ----------
        dim : {2, 3}
            Dimension
        transform_type : {'rigid', 'rigid+scale', 'affine'}, default='rigid'
            Transformation type
        """
        super().__init__()
        self.dim = dim
        self.transform_type = transform_type
        self.basis = None

    def build(self, input_shape):
        group = self.transform_type
        if group not in groups:
            if group == 'rigid':
                group = 'SE'
            elif group == 'rigid+scale':
                group = 'CSO'
            else:
                group = 'Aff+'
        basis = affine_basis(self.dim, group)
        self.basis = tf.Variable(initial_value=basis)

    def call(self, affine):
        """

        Parameters
        ----------
        affine : (batch, dim+1, dim+1) tensor
            Batch of (square) affine matrices
        """
        if isinstance(affine, (list, tuple)):
            prm = affine[0]
        return affine_log(affine, self.basis)


def affine_exp(prm, basis):
    r"""Reconstruct an affine matrix from its Lie parameters.

    Parameters
    ----------
    prm : (..., F) tensor
        Parameters in the Lie algebra.

    basis : (F, D+1, D+1) tensor
        Basis of the Lie algebras.

    Returns
    -------
    mat : (..., D+1, D+1) tensor
        Reconstructed affine matrix.

    """

    # Check length
    if prm.shape[-1] != basis.shape[0]:
        raise ValueError('Number of parameters and number of bases do '
                         'not match. Got {} and {}'
                         .format(prm.shape[-1], basis.shape[0]))

    # Reconstruct the log-matrix and exponentiate
    return tf.linalg.expm(tf.reduce_sum(basis*prm[..., None, None], axis=-3))


def affine_log(mat, basis):
    """Recover Lie parameters from an affine matrix.

    Parameters
    ----------
    mat : (..., D+1, D+1) tensor
        Reconstructed affine matrix.
    basis : (F, D+1, D+1) tensor
        Basis of the Lie algebras.

    Returns
    -------
    prm : (..., F) tensor
        Parameters in the Lie algebra.
    """
    dtype = mat.dtype
    mat = tf.cast(mat, tf.complex128)
    mat = tf.linalg.logm(mat)
    mat = tf.cast(tf.math.real(mat), dtype)
    prm = mdot(mat[..., None, :, :], basis)
    prm = tf.cast(tf.math.real(prm), dtype)
    return prm


def mdot(a, b):
    """Compute the Frobenius inner product of two matrices

    Parameters
    ----------
    a : (..., N, M) tensor
        Left matrix
    b : (..., N, M) tensor
        Right matrix

    Returns
    -------
    dot : (...) tensor
        Matrix inner product

    References
    ----------
    ..[1] https://en.wikipedia.org/wiki/Frobenius_inner_product

    """
    return tf.linalg.trace(tf.linalg.matmul(a, b, adjoint_a=True))


groups = ('T', 'SO', 'SE', 'D', 'CSO', 'SL', 'GL+', 'Aff+')


def affine_basis(dim, group='SE', dtype=tf.float32):
    """Generate a basis set for the algebra of some (Lie) group of matrices.

    The basis is returned in homogeneous coordinates, even if
    the group does not require translations. To extract the linear
    part of the basis: lin = basis[:-1, :-1].

    This function focuses on 'classic' Lie groups. Note that, while it
    is commonly used in registration software, we do not have a
    "9-parameter affine" (translations + rotations + zooms),
    because such transforms do not form a group; that is, their inverse
    may contain shears.

    Parameters
    ----------
    dim : {2, 3}
        Dimension
    group : {'T', 'SO', 'SE', 'D', 'CSO', 'SL', 'GL+', 'Aff+'}, default='SE'
        Group that should be encoded by the basis set:
            * 'T'   : Translations
            * 'SO'  : Special Orthogonal (rotations)
            * 'SE'  : Special Euclidean (translations + rotations)
            * 'D'   : Dilations (translations + isotropic scalings)
            * 'CSO' : Conformal Special Orthogonal
                      (translations + rotations + isotropic scalings)
            * 'SL'  : Special Linear (rotations + isovolumic zooms + shears)
            * 'GL+' : General Linear [det>0] (rotations + zooms + shears)
            * 'Aff+': Affine [det>0] (translations + rotations + zooms + shears)
    dtype : tf.dtype, default=tf.float32
        Data type of the returned array

    Returns
    -------
    basis : (F, dim+1, dim+1) tensor[dtype]
        Basis set, where ``F`` is the number of basis functions.

    """

    if dim not in (2, 3):
        raise ValueError('Dimension must be 2 or 3. Got {}.'.format(dim))
    if group not in ('T', 'SO', 'SE', 'D', 'CSO', 'SL', 'GL+', 'Aff+'):
        raise ValueError('Unknown group {}.'.format(group))

    if group == 'T':
        return affine_subbasis(dim, 'T', dtype=dtype)
    elif group == 'SO':
        return affine_subbasis(dim, 'R', dtype=dtype)
    elif group == 'SE':
        return tf.concat((affine_subbasis(dim, 'T', dtype=dtype),
                          affine_subbasis(dim, 'R', dtype=dtype)), axis=0)
    elif group == 'D':
        return tf.concat((affine_subbasis(dim, 'T', dtype=dtype),
                          affine_subbasis(dim, 'I', dtype=dtype)), axis=0)
    elif group == 'CSO':
        return tf.concat((affine_subbasis(dim, 'T', dtype=dtype),
                          affine_subbasis(dim, 'R', dtype=dtype),
                          affine_subbasis(dim, 'I', dtype=dtype)), axis=0)
    elif group == 'SL':
        return tf.concat((affine_subbasis(dim, 'R', dtype=dtype),
                          affine_subbasis(dim, 'Z0', dtype=dtype),
                          affine_subbasis(dim, 'S', dtype=dtype)), axis=0)
    elif group == 'GL+':
        return tf.concat((affine_subbasis(dim, 'R', dtype=dtype),
                          affine_subbasis(dim, 'Z', dtype=dtype),
                          affine_subbasis(dim, 'S', dtype=dtype)), axis=0)
    elif group == 'Aff+':
        return tf.concat((affine_subbasis(dim, 'T', dtype=dtype),
                          affine_subbasis(dim, 'R', dtype=dtype),
                          affine_subbasis(dim, 'Z', dtype=dtype),
                          affine_subbasis(dim, 'S', dtype=dtype)), axis=0)
    else:
        assert False


def affine_subbasis(dim, mode, dtype=tf.float32):
    """Generate a basis set for the algebra of some (Lie) group of matrices.

    The basis is returned in homogeneous coordinates, even if
    the group required does not require translations. To extract the
    linear part of the basis: lin = basis[..., :-1, :-1].

    This function focuses on very simple (and coherent) groups.

    Note that shears generated by the 'S' basis are not exactly the same
    as classical shears ('SC'). Setting one classical shear parameter to
    a non-zero value generally applies a gradient of translations along
    a direction:
    + -- +         + -- +
    |    |   =>   /    /
    + -- +       + -- +
    Setting one Lie shear parameter to a non zero value is more alike
    to performing an expansion in one (diagonal) direction and a
    contraction in the orthogonal (diagonal) direction. It is a bit
    harder to draw in ascii, but it can also be seen as a horizontal
    shear followed by a vertical shear.

    Parameters
    ----------
    dim : {2, 3}
        Dimension

    mode : {'T', 'R', 'Z', 'Z0', 'I', 'S', 'SC'}
        Group that should be encoded by the basis set:
            * 'T'   : Translations                     [dim]
            * 'R'   : Rotations                        [dim*(dim-1)//2]
            * 'Z'   : Zooms (= anisotropic scalings)   [dim]
            * 'Z0'  : Isovolumic scalings              [dim-1]
            * 'I'   : Isotropic scalings               [1]
            * 'S'   : Shears (symmetric)               [dim*(dim-1)//2]
            * 'SC'  : Shears (classic)                 [dim*(dim-1)//2]

    dtype : tf.dtype, default=tf.float32
        Data type of the returned array.

    Returns
    -------
    basis : (F, dim+1, dim+1) tensor[dtype]
        Basis set, where ``F`` is the number of basis functions.

    """

    if dim not in (2, 3):
        raise ValueError('Dimension must be 2 or 3. Got {}.'.format(dim))
    if mode not in ('T', 'Z', 'Z0', 'I', 'R', 'S', 'SC'):
        raise ValueError('Unknown group {}.'.format(mode))

    if mode == 'T':
        if dim == 2:
            basis = [
                [[0, 0, 1],
                 [0, 0, 0],
                 [0, 0, 0]],
                [[0, 0, 0],
                 [0, 0, 1],
                 [0, 0, 0]],
            ]
        else:
            basis = [
                [[0, 0, 0, 1],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0]],
                [[0, 0, 0, 0],
                 [0, 0, 0, 1],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0]],
                [[0, 0, 0, 0],
                 [0, 0, 0, 0],
                 [0, 0, 0, 1],
                 [0, 0, 0, 0]],
            ]

    elif mode == 'Z':
        if dim == 2:
            basis = [
                [[1, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0]],
                [[0, 0, 0],
                 [0, 1, 0],
                 [0, 0, 0]],
            ]
        else:
            basis = [
                [[1, 0, 0, 0],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0]],
                [[0, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0]],
                [[0, 0, 0, 0],
                 [0, 0, 0, 0],
                 [0, 0, 1, 0],
                 [0, 0, 0, 0]],
            ]

    elif mode == 'Z0':
        if dim == 2:
            a = 1/math.sqrt(2)
            basis = [
                [[a, 0, 0],
                 [0,-a, 0],
                 [0, 0, 0]],
            ]
        else:
            a = 1/math.sqrt(2)
            b = 1/math.sqrt(6)
            basis = [
                [[a, 0, 0, 0],
                 [0, 0, 0, 0],
                 [0, 0,-a, 0],
                 [0, 0, 0, 0]],
                [[b,   0, 0, 0],
                 [0,-2*b, 0, 0],
                 [0,   0, b, 0],
                 [0,   0, 0, 0]],
            ]

    elif mode == 'I':
        if dim == 2:
            a = 1/math.sqrt(2)
            basis = [
                [[a, 0, 0],
                 [0, a, 0],
                 [0, 0, 0]],
            ]
        else:
            a = 1/math.sqrt(3)
            basis = [
                [[a, 0, 0, 0],
                 [0, a, 0, 0],
                 [0, 0, a, 0],
                 [0, 0, 0, 0]],
            ]

    elif mode == 'R':
        a = 1 / math.sqrt(2)
        if dim == 2:
            basis = [
                [[ 0, a, 0],
                 [-a, 0, 0],
                 [ 0, 0, 0]],
            ]
        else:
            basis = [
                [[ 0, a, 0, 0],
                 [-a, 0, 0, 0],
                 [ 0, 0, 0, 0],
                 [ 0, 0, 0, 0]],
                [[ 0, 0, a, 0],
                 [ 0, 0, 0, 0],
                 [-a, 0, 0, 0],
                 [ 0, 0, 0, 0]],
                [[0,  0, 0, 0],
                 [0,  0, a, 0],
                 [0, -a, 0, 0],
                 [0,  0, 0, 0]],
            ]

    elif mode == 'S':
        a = 1 / math.sqrt(2)
        if dim == 2:
            basis = [
                [[0, a, 0],
                 [a, 0, 0],
                 [0, 0, 0]],
            ]
        else:
            basis = [
                [[0, a, 0, 0],
                 [a, 0, 0, 0],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0]],
                [[0, 0, a, 0],
                 [0, 0, 0, 0],
                 [a, 0, 0, 0],
                 [0, 0, 0, 0]],
                [[0, 0, 0, 0],
                 [0, 0, a, 0],
                 [0, a, 0, 0],
                 [0, 0, 0, 0]],
            ]

    elif mode == 'SC':
        if dim == 2:
            basis = [
                [[0, 1, 0],
                 [0, 0, 0],
                 [0, 0, 0]],
            ]
        else:
            basis = [
                [[0, 1, 0, 0],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0]],
                [[0, 0, 1, 0],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0]],
                [[0, 0, 0, 0],
                 [0, 0, 1, 0],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0]],
            ]

    else:
        # We should never reach this (a test was performed earlier)
        assert False

    return tf.convert_to_tensor(basis, dtype=dtype)
