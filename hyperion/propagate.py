"""Implementation of the photon propagation code."""
import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from jax.lax import cond, fori_loop, while_loop
from scipy.integrate import quad

from .constants import Constants
from .utils import rotate_to_new_direc


def sph_to_cart(theta, phi=0, r=1):
    """Transform spherical to cartesian coordinates."""
    x = r * jnp.sin(theta) * jnp.cos(phi)
    y = r * jnp.sin(theta) * jnp.sin(phi)
    z = r * jnp.cos(theta)

    return jnp.array([x, y, z])


def photon_sphere_intersection(
    photon_x, photon_p, target_x, target_r, step_size, dtype=jnp.float64
):
    """
    Calculate intersection.

    Given a photon origin, a photon direction, a step size, a target location and a target radius,
    calculate whether the photon intersects the target and the intrsection point.

    Parameters:
        photon_x: float[3]
        photon_p: float[3]
        step_size: float

    Returns:
        tuple(bool, float[3])
            True and intersection position if intersected.
    """
    p_normed = jnp.asarray(photon_p, dtype=dtype)  # assume normed

    a = jnp.dot(p_normed, (photon_x - target_x))
    b = a ** 2 - (jnp.linalg.norm(photon_x - target_x) ** 2 - target_r ** 2)
    # Distance of of the intersection point along the line
    d = -a - jnp.sqrt(b)

    isected = (b >= 0) & (d > 0) & (d < step_size)

    # need to check intersection here, otherwise nan-gradients (sqrt(b) if b < 0)
    result = cond(
        isected,
        lambda _: (True, photon_x + d * p_normed),
        lambda _: (False, jnp.ones(3) * 1e8),
        0,
    )

    return result


def make_photon_sphere_intersection_func(target_x, target_r, dtype=jnp.float64):
    """
    Make a function that calculates the intersection of a photon path with a sphere.

    Parameters:
        target_x: float[3]
        target_r: float
    """
    target_x = jnp.asarray(target_x, dtype=dtype)
    target_r = dtype(target_r)

    return functools.partial(
        photon_sphere_intersection, target_x=target_x, target_r=target_r, dtype=dtype
    )


def make_multi_photon_sphere_intersection_func(target_x, target_r, dtype=jnp.float64):
    """
    Make a function that calculates the intersection of a photon path with multiple spheres.

    Parameters:
        target_x: float[N, 3]
        target_r: float[N]
    """
    target_x = jnp.asarray(target_x, dtype=dtype)
    target_r = jnp.asarray(target_r, dtype=dtype)

    isec_func_v = jax.vmap(photon_sphere_intersection, in_axes=[None, None, 0, 0, None])

    def f(photon_x, photon_p, step_size):
        isecs = isec_func_v(photon_x, photon_p, target_x, target_r, step_size)
        any_isec = jnp.any(isecs[0])

        return cond(
            any_isec,
            lambda _: (True, isecs[1].at[jnp.argsort(isecs[0])[-1]].get()),
            lambda _: (False, jnp.ones(3, dtype=dtype) * 1e8),
            0,
        )

    return f


def make_photon_spherical_shell_intersection(
    shell_center, shell_radius, dtype=jnp.float64
):
    shell_center = jnp.asarray(shell_center, dtype=dtype)
    shell_radius = dtype(shell_radius)

    def photon_spherical_shell_intersection(photon_x, photon_p, step_size):
        p_normed = jnp.asarray(photon_p, dtype=dtype)  # assume normed

        a = jnp.dot(p_normed, (photon_x - shell_center))
        b = a ** 2 - (jnp.linalg.norm(photon_x - shell_center) ** 2 - shell_radius ** 2)

        # Distance of of the intersection point along the line
        d = -a + jnp.sqrt(b)

        isected = (b >= 0) & (d > 0) & (d < step_size)

        # need to check intersection here, otherwise nan-gradients (sqrt(b) if b < 0)
        result = cond(
            isected,
            lambda _: (True, photon_x + d * p_normed),
            lambda _: (False, jnp.ones(3, dtype=dtype) * 1e8),
            0,
        )

        return result

    return photon_spherical_shell_intersection


def make_photon_circle_intersection(
    circle_center, circle_normal, circle_r, dtype=jnp.float64
):

    circle_center = jnp.asarray(circle_center, dtype=dtype)
    circle_normal = jnp.asarray(circle_normal, dtype=dtype)
    circle_r = dtype(circle_r)

    def photon_circle_intersection(photon_x, photon_p, step_size):
        """
        Intersection of line and plane.

        iven a photon origin, a photon direction, a step size, a target location and a target radius,
        calculate whether the photon intersects the target and the intersection point.

        Parameters:
            photon_x: float[3]
            photon_p: float[3]
            step_size: float

        Returns:
            tuple(bool, float[3])
                True and intersection position if intersected.
        """
        # assume plane normal vector is e_z

        photon_p = jnp.asarray(photon_p, dtype=dtype)
        p_n = jnp.dot(photon_p, circle_normal)
        d = jnp.where(
            p_n == 0,
            jnp.dot((circle_center - photon_x), circle_normal),
            jnp.dot((circle_center - photon_x), circle_normal) / p_n,
        )

        isec_p = photon_x + d * photon_p

        dist_in_plane = jnp.linalg.norm(circle_center - isec_p)

        result = jax.lax.cond(
            (d > 0) & (d <= step_size) & (dist_in_plane < circle_r),
            lambda _: (True, isec_p),
            lambda _: (False, jnp.ones(3, dtype=dtype) * 1e8),
            None,
        )
        return result

    return photon_circle_intersection


def frank_tamm(wavelength, ref_index_func):
    """Frank Tamm Formula."""
    return (
        4
        * np.pi ** 2
        * Constants.BaseConstants.e ** 2
        / (
            Constants.BaseConstants.h
            * Constants.BaseConstants.c_vac
            * (wavelength / 1e9) ** 2
        )
        * (1 - 1 / ref_index_func(wavelength) ** 2)
    )


def make_cherenkov_spectral_sampling_func(wl_range, ref_index_func, dtype=jnp.float64):
    """
    Make a sampling function that samples from the Frank-Tamm formula in a given wavelength range.

    Parameters:
        wl_range: Tuple
            lower and upper wavelength range
        ref_index_func: function
            Function returning wavelength dependent refractive index

    """
    wls = np.linspace(wl_range[0], wl_range[1], 1000)

    integral = lambda upper: quad(  # noqa E731
        functools.partial(frank_tamm, ref_index_func=ref_index_func), wl_range[0], upper
    )[0]
    norm = integral(wl_range[-1])
    poly_pars = jnp.asarray(
        np.polyfit(np.vectorize(integral)(wls) / norm, wls, 10), dtype=dtype
    )

    def sampling_func(rng_key):
        uni = random.uniform(rng_key, dtype=dtype)
        return jnp.polyval(poly_pars, uni)

    return sampling_func


def calc_new_direction(keys, old_dir, scattering_function):
    """
    Calculate new direction after sampling a scattering angle.

    Scattering is calculated in a reference frame local
    to the photon (e_z) and then rotated back to the global coordinate system.
    """

    theta = scattering_function(keys[0])
    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)

    phi = random.uniform(keys[1], minval=0, maxval=2 * np.pi)
    cos_phi = jnp.cos(phi)
    sin_phi = jnp.sin(phi)

    px, py, pz = old_dir

    is_para_z = jnp.abs(pz) == 1

    new_dir = cond(
        is_para_z,
        lambda _: jnp.array(
            [
                sin_theta * cos_phi,
                jnp.sign(pz) * sin_theta * sin_phi,
                jnp.sign(pz) * cos_theta,
            ]
        ),
        lambda _: jnp.array(
            [
                (px * cos_theta)
                + (
                    (sin_theta * (px * pz * cos_phi - py * sin_phi))
                    / (jnp.sqrt(1.0 - pz ** 2))
                ),
                (py * cos_theta)
                + (
                    (sin_theta * (py * pz * cos_phi + px * sin_phi))
                    / (jnp.sqrt(1.0 - pz ** 2))
                ),
                (pz * cos_theta) - (sin_theta * cos_phi * jnp.sqrt(1.0 - pz ** 2)),
            ]
        ),
        None,
    )

    # Need this for numerical stability?
    new_dir = new_dir / jnp.linalg.norm(new_dir)

    return new_dir


def make_step_function(
    intersection_f,
    scattering_function,
    scattering_length_function,
    ref_index_func,
    dtype=jnp.float64,
):
    """
    Make a photon step function object.

    Returns a function f(photon_state, key) that performs a photon step
    and returns the new photon state.

    Parameters:
        intersection_f: function
            function used to calculate the intersection
        scattering_function: function
            rng function drawing angles from scattering function
        scattering_length_function: function
            function that returns scattering length as function of wavelength
        ref_index_func: function
            function that returns the refractive index as function of wavelength
    """

    def step(photon_state, rng_key):
        """Single photon step."""
        pos = photon_state["pos"]
        dir = photon_state["dir"]
        time = photon_state["time"]
        isec = photon_state["isec"]
        stepcnt = photon_state["stepcnt"]
        wavelength = photon_state["wavelength"]

        k1, k2, k3, k4 = random.split(rng_key, 4)

        sca_coeff = 1 / scattering_length_function(wavelength)
        c_medium = (
            Constants.BaseConstants.c_vac * 1e-9 / ref_index_func(wavelength)
        )  # m/ns

        eta = random.uniform(k1)
        step_size = -jnp.log(eta) / sca_coeff

        dstep = step_size * dir
        new_pos = jnp.asarray(pos + dstep, dtype=dtype)
        new_time = dtype(time + step_size / c_medium)

        # Calculate intersection
        isec, isec_pos = intersection_f(
            photon_x=pos,
            photon_p=dir,
            step_size=step_size,
        )

        isec_time = dtype(time + jnp.linalg.norm(pos - isec_pos) / c_medium)

        # If intersected, set position to intersection position
        new_pos = cond(
            isec, lambda args: args[0], lambda args: args[1], (isec_pos, new_pos)
        )

        # If intersected set time to intersection time
        new_time = cond(
            isec,
            lambda args: args[0],
            lambda args: args[1],
            (isec_time, new_time),
        )

        # If intersected, keep previous direction
        new_dir = cond(
            isec,
            lambda args: args[1],
            lambda args: calc_new_direction(args[0], args[1], scattering_function),
            ([k2, k3], dir),
        )

        stepcnt = cond(isec, lambda s: s, lambda s: s + 1, stepcnt)

        new_photon_state = {
            "pos": new_pos,
            "dir": new_dir,
            "time": new_time,
            "isec": isec,
            "stepcnt": stepcnt,
            "wavelength": wavelength,
        }

        return new_photon_state, k4

    return step


def unpack_args(f):
    """Wrap a function by unpacking a single argument tuple."""

    def _f(args):
        return f(*args)

    return _f


@functools.partial(
    jax.profiler.annotate_function, name="initialize_direction_isotropic"
)
def initialize_direction_isotropic(rng_key):
    """Draw direction uniformly on a sphere."""
    k1, k2 = random.split(rng_key, 2)
    theta = jnp.arccos(random.uniform(k1, minval=-1, maxval=1))
    phi = random.uniform(k2, minval=0, maxval=2 * np.pi)
    direction = sph_to_cart(theta, phi, r=1)

    return direction


def initialize_direction_led(rng_key):
    k1, k2 = random.split(rng_key, 2)
    theta = jnp.arcsin(random.uniform(k1))
    phi = random.uniform(k2, minval=0, maxval=2 * np.pi)

    direction = sph_to_cart(theta, phi, r=1)

    return direction


def make_initialize_direction_laser(direction):
    def initialize_direction_laser(rng_key):
        """Return e_z."""
        return direction

    return initialize_direction_laser


def make_initialize_position_sphere(sphere_pos, sphere_radius):
    def initialize_position_sphere(rng_key):
        direc_vec = initialize_direction_isotropic(rng_key)

        pos = sphere_pos + direc_vec * sphere_radius

        return pos

    return initialize_position_sphere


def make_monochromatic_initializer(wavelength):
    """Make a monochromatic initializer function."""

    def initialize_monochromatic(rng_key):
        return wavelength

    return initialize_monochromatic


wl_mono_400nm_init = make_monochromatic_initializer(400)


def make_fixed_pos_time_initializer(
    initial_pos, initial_time, dir_init, wavelength_init, dtype=jnp.float64
):
    """
    Initialize with fixed position and time and sample for direction and wavelength.

    initial_pos: float[3]
        Position vector of the emitter
    initial_time: float
        Emitter time
    dir_init: function
        Emission direction initializer
    wavelength_init: function
        wavelength initializer
    """

    def init(rng_key):
        k1, k2 = random.split(rng_key, 2)

        # Set initial photon state
        initial_photon_state = {
            "pos": jnp.asarray(initial_pos, dtype=dtype),
            "dir": dir_init(k1),
            "time": dtype(initial_time),
            "isec": False,
            "stepcnt": jnp.int32(0),
            "wavelength": wavelength_init(k2),
        }
        return initial_photon_state

    return init


def make_fixed_time_initializer(
    initial_time, pos_init, dir_init, wavelength_init, dtype=jnp.float64
):
    """
    Initialize with a fixed time, sample for position, direction and wavelength.

    initial_pos: float[3]
        Position vector of the emitter
    initial_time: float
        Emitter time
    pos_init: function
        Position initializer
    dir_init: function
        Emission direction initializer
    wavelength_init: function
        wavelength initializer
    """

    def init(rng_key):
        k1, k2, k3 = random.split(rng_key, 3)

        # Set initial photon state
        initial_photon_state = {
            "pos": jnp.asarray(pos_init(k1), dtype=dtype),
            "dir": dir_init(k2),
            "time": dtype(initial_time),
            "isec": False,
            "stepcnt": jnp.int32(0),
            "wavelength": wavelength_init(k3),
        }
        return initial_photon_state

    return init


def make_track_segment_fixed_time_pos_dir_initializer(
    initial_time,
    track_pos,
    track_dir,
    wavelength_init,
    phase_velo_func,
    segment_length,
    dtype=jnp.float64,
):
    """
    Initialize a track segment with fixed time, position and direction.

    The photon emission position is sampled uniformly along the track (while adjusting the time).
    Emission angle is calculated from the wavelength.

    initial_time: float
        Track starting time
    track_pos: float[3]
        Track starting position
    track_dir: float[3]
        Track direction
    wavelength_init: function
        wavelength initializer
    phase_velo_func: function
        function that returns the phase velocity as function of wavelength
    segment_length: float
        length of the segment
    """

    initial_time = dtype(initial_time)
    track_pos = jnp.asarray(track_pos, dtype=dtype)
    track_dir = jnp.asarray(track_dir, dtype=dtype)

    def init(rng_key):
        k1, k2, k3 = random.split(rng_key, 3)

        # sample position along track
        dist_along = random.uniform(k1, minval=0, maxval=segment_length)
        time_along = initial_time + dist_along / Constants.BaseConstants.c_vac
        position = track_pos + dist_along * track_dir

        # sample wavelength
        wl = wavelength_init(k2)

        cherenkov_angle_theta = jnp.arccos(1 / phase_velo_func(wl))
        phi_angle = random.uniform(k3, minval=0, maxval=2 * np.pi)

        photon_rel_dir = sph_to_cart(cherenkov_angle_theta, phi_angle)

        photon_dir = rotate_to_new_direc(
            jnp.asarray([0, 0, 1.0]), track_dir, photon_rel_dir
        )

        # Set initial photon state
        initial_photon_state = {
            "pos": position,
            "dir": photon_dir,
            "time": time_along,
            "isec": False,
            "stepcnt": jnp.int32(0),
            "wavelength": wl,
        }
        return initial_photon_state

    return init


def make_loop_until_isec_or_maxtime(max_time):
    """Make function that will call the step_function until either the photon intersetcs or max_time is reached."""

    def loop_until_isec_or_maxtime(step_function, initial_photon_state, rng_key):
        final_photon_state, _ = while_loop(
            lambda args: (args[0]["isec"] == False)  # noqa: E712
            & (args[0]["time"] < max_time),
            unpack_args(step_function),
            (initial_photon_state, rng_key),
        )
        return final_photon_state

    return loop_until_isec_or_maxtime


def make_loop_for_n_steps(n_steps):
    """Make function that calls step_function n_steps times."""

    def loop_for_nsteps(step_function, initial_photon_state, rng_key):
        def noop_if_not_alive(state, rng_key):
            out = cond(
                state["isec"],
                lambda args: args,
                unpack_args(step_function),
                (state, rng_key),
            )
            return out

        final_photon_state, _ = fori_loop(
            0,
            n_steps,
            lambda i, args: unpack_args(noop_if_not_alive)(args),
            (initial_photon_state, rng_key),
        )
        return final_photon_state

    return loop_for_nsteps


def make_photon_trajectory_fun(
    step_function,
    photon_init_function,
    loop_func,
):
    """
    Make a photon trajectory function.

    This function calls the photon step function multiple times until
    some termination condition is reached (defined by `stepping_mode`)

    step_function: function
        Function that updates the photon state

    photon_init_function: function
        Function that returns the initial photon state

    loop_func: function
        Looping function that calls the photon step function.
    """

    def make_steps(key):
        """
        Make a function that steps a photon until it either intersects or  max length is reached.

        Parameters:
            key: PRNGKey

        Returns:
            photon_state: dict
                Final photon state
        """
        k1, k2 = random.split(key, 2)

        # Set initial photon state
        initial_photon_state = photon_init_function(k1)

        final_photon_state = loop_func(step_function, initial_photon_state, k2)

        return initial_photon_state, final_photon_state

    return make_steps


def collect_hits(traj_func, nphotons, nsims, seed=0, sim_limit=1e7):
    """Run photon prop multiple times and collect hits."""
    key = random.PRNGKey(seed)
    isec_times = []
    em_thetas = []
    ar_thetas = []
    stepss = []
    nphotons = int(nphotons)
    isec_poss = []
    wavelengths = []

    total_detected_photons = 0
    sims_cnt = 0

    for i in range(nsims):
        key, subkey = random.split(key)
        initial_state, final_state = traj_func(random.split(key, num=nphotons))

        isecs = final_state["isec"]

        isec_times.append(np.asarray(final_state["time"][isecs]))
        stepss.append(np.asarray(final_state["stepcnt"][isecs]))
        ar_thetas.append(np.asarray(jnp.arccos(final_state["dir"][isecs, 2])))
        em_thetas.append(np.asarray(jnp.arccos(initial_state["dir"][isecs, 2])))
        isec_poss.append(np.asarray(final_state["pos"][isecs]))
        wavelengths.append(np.asarray(final_state["wavelength"][isecs]))

        sims_cnt = i
        total_detected_photons += jnp.sum(isecs)
        if sim_limit is not None and total_detected_photons > sim_limit:
            break

    isec_times = np.concatenate(isec_times)
    ar_thetas = np.concatenate(ar_thetas)
    em_thetas = np.concatenate(em_thetas)
    stepss = np.concatenate(stepss)
    isec_poss = np.vstack(isec_poss)
    wavelengths = np.concatenate(wavelengths)

    return isec_times, ar_thetas, em_thetas, stepss, isec_poss, sims_cnt, wavelengths
