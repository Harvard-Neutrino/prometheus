# Prometheus

Welcome to Prometheus, an open-source neutrino telescope simulation.

|              |                                                  |
|--------------|--------------------------------------------------|
|Repository    | <https://github.com/Harvard-Neutrino/prometheus> |
|ArXiv paper   | <http://arxiv.org/abs/2304.1452>                 |
|Documentation | <https://harvard-neutrino.github.io/prometheus/> |

## Summary

Prometheus is a Python-based package for simulating neutrino telescopes. It balances ease of use with performance and supports simulations with arbitrary detector geometries deployed in ice or water.

Prometheus simulates neutrino interactions in the volume surrounding the detector, computes the light yield of hadronic showers and the outgoing leptons, propagates photons in the medium, and records their arrival times and positions in user-defined regions. Events are then serialized to Parquet files, a compact and interoperable format that enables efficient access for downstream analysis.

## Terms of Use

Prometheus is open-source. You are free to copy, modify, and distribute it with attribution under the terms of the GNU Lesser General Public License. See the [LICENSE](./LICENSE.md) file for details.

## Citation

Please cite Prometheus using this entry:

```bibtex
@article{Lazar:2023rol,
    author = {Lazar, Jeffrey and Meighen-Berger, Stephan and Haack, Christian and Kim, David and Giner, Santiago and Arg{\"u}elles, Carlos A.},
    title = "{Prometheus: An open-source neutrino telescope simulation}",
    eprint = "2304.14526",
    archivePrefix = "arXiv",
    primaryClass = "hep-ex",
    doi = "10.1016/j.cpc.2024.109298",
    journal = "Comput. Phys. Commun.",
    volume = "304",
    pages = "109298",
    year = "2024"
}
```

Please also consider citing the packages that Prometheus uses internally: LeptonInjector, PROPOSAL, ppc, and LeptonWeighter with the following citations:

<details>
  <summary>LeptonInjector and LeptonWeighter</summary>

  ```bibtex
  @article{IceCube:2020tcq,
      author = "Abbasi, R. and others",
      collaboration = "IceCube",
      title = "{LeptonInjector and LeptonWeighter: A neutrino event generator and weighter for neutrino observatories}",
      eprint = "2012.10449",
      archivePrefix = "arXiv",
      primaryClass = "physics.comp-ph",
      doi = "10.1016/j.cpc.2021.108018",
      journal = "Comput. Phys. Commun.",
      volume = "266",
      pages = "108018",
      year = "2021"
  }
  ```
</details>

<details>
  <summary>PROPOSAL</summary>

  ```bibtex
  @article{koehne2013proposal,
      title     = {PROPOSAL: A tool for propagation of charged leptons},
      author    = {Koehne, Jan-Hendrik and
                  Frantzen, Katharina and
                  Schmitz, Martin and
                  Fuchs, Tomasz and
                  Rhode, Wolfgang and
                  Chirkin, Dmitry and
                  Tjus, J Becker},
      journal   = {Computer Physics Communications},
      volume    = {184},
      number    = {9},
      pages     = {2070--2090},
      year      = {2013},
      doi       = {10.1016/j.cpc.2013.04.001}
  }
  ```

</details>

<details>
  <summary>ppc</summary>

  ```bibtex
  @misc{chirkin2022kpl,
      author = {D. Chirkin},
      title = {ppc},
      year = {2022},
      publisher = {GitHub},
      journal = {GitHub repository},
      howpublished = {\url{https://github.com/icecube/ppc}},
      commit = {30ea4ada13fbcf996c58a3eb3f0b1358be716fc8}
  }

  ```

</details>

## Contributing

We always appreciate contributions that can make Prometheus better.

Read our [contribution guidelines](./CONTRIBUTING.md#issues) to learn more about how you can contribute.

## Getting Help

Questions or setup/usage issues? Create [a discussion](https://github.com/Harvard-Neutrino/prometheus/discussions).

Found a bug or want to suggest a change? [Open an issue](https://github.com/Harvard-Neutrino/prometheus/issues/new/choose).

More information on opening issues and getting help is available in our [contribution guidelines](./CONTRIBUTING.md#).
