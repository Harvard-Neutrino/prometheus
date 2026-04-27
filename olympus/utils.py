import jax
import jax.numpy as jnp
from jax.lax import Precision


def rotate_to_new_direc(old_dir, new_dir, operand):
    """Rotate a vector from one reference direction to another using Rodrigues' formula.

    Parameters
    ----------
    old_dir : jnp.ndarray
        Original reference direction (unit vector).
    new_dir : jnp.ndarray
        Target reference direction (unit vector).
    operand : jnp.ndarray
        Vector to rotate.

    Returns
    -------
    v_rot : jnp.ndarray
        Rotated vector.
    """

    def _rotate(operand):

        axis = jnp.cross(old_dir, new_dir)
        axis /= jnp.linalg.norm(axis)

        theta = jnp.arccos(jnp.dot(old_dir, new_dir, precision=Precision.HIGHEST))

        # Rodrigues' rotation formula

        v_rot = (
            operand * jnp.cos(theta)
            + jnp.cross(axis, operand) * jnp.sin(theta)
            + axis * jnp.dot(axis, operand, precision=Precision.HIGHEST) * (1 - jnp.cos(theta))
        )
        return v_rot

    v_rot = jax.lax.cond(jnp.all(old_dir == new_dir), lambda op: op, _rotate, operand)

    return v_rot


rotate_to_new_direc_v = jax.jit(jax.vmap(rotate_to_new_direc, in_axes=[None, None, 0]))
