import proposal as pp

args = {
    "particle_def": pp.particle.MuMinusDef(),
    "target": pp.medium.Water(),
    "interpolate": True,
    "cuts": pp.EnergyCutSettings(500, 1, False),
}

cross = pp.crosssection.make_std_crosssection(**args)  # use the standard crosssections
collection = pp.PropagationUtilityCollection()

collection.displacement = pp.make_displacement(cross, True)
collection.interaction = pp.make_interaction(cross, True)
collection.time = pp.make_time(cross, args["particle_def"], True)

utility = pp.PropagationUtility(collection=collection)

detector = pp.geometry.Sphere(pp.Cartesian3D(0, 0, 0), 1e20)
density_distr = pp.density_distribution.density_homogeneous(args["target"].mass_density)
prop = pp.Propagator(args["particle_def"], [(detector, utility, density_distr)])
