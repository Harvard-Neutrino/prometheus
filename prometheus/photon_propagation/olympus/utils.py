import jax.numpy as jnp
from jax.lax import Precision
import jax


def rotate_to_new_direc(old_dir, new_dir, operand):
    def _rotate(operand):

        axis = jnp.cross(old_dir, new_dir)
        axis /= jnp.linalg.norm(axis)

        theta = jnp.arccos(jnp.dot(old_dir, new_dir, precision=Precision.HIGHEST))

        # Rodrigues' rotation formula

        v_rot = (
            operand * jnp.cos(theta)
            + jnp.cross(axis, operand) * jnp.sin(theta)
            + axis
            * jnp.dot(axis, operand, precision=Precision.HIGHEST)
            * (1 - jnp.cos(theta))
        )
        return v_rot

    v_rot = jax.lax.cond(jnp.all(old_dir == new_dir), lambda op: op, _rotate, operand)

    return v_rot


rotate_to_new_direc_v = jax.jit(jax.vmap(rotate_to_new_direc, in_axes=[None, None, 0]))
