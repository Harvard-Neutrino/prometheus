import setuptools

version = "0.0.0"

setuptools.setup(
    name="prometheus", 
    version=version,
    author="J. Lazar, et al.",
    author_email="jlazar@icecube.wisc.edu",
    description="Code for simulating neutrino telescopes",
    #long_description=long_message,
    #long_description_content_type="text/markdown",
    url="https://github.com/Harvard-Neutrino/prometheus",
    packages=setuptools.find_packages(),
    install_requires=[
	"jax"
	"pyyaml"
        "numpy>=1.16.6",
        "awkward>=1.8.0",
        "scipy>=1.2.3",
        "pyarrow>=7.0.0",
        "json>=2.0.9",
        # Should we make this optional
        "tqdm>=4.52.0",
        "jax>=0.2.21",
        "proposal>=6.1.6",
        "h5py",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)"
    ],
    python_requires='>=3.7',
