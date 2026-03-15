"""
Prometheus smoke test — minimal end-to-end validation.

Runs 5 events with the demo_ice.geo detector and the PPC CPU propagator,
then verifies that the output parquet file has the expected structure.

Exit code 0 on success, 1 on failure.
"""

import os
import sys
import shutil
import tempfile
import traceback

# ------------------------------------------------------------
# Paths derived from this script's location (all absolute)
# ------------------------------------------------------------
REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
RESOURCE_DIR = os.path.join(REPO_DIR, "resources")

# ------------------------------------------------------------
# Expected fields
# ------------------------------------------------------------
EXPECTED_MC_TRUTH_FIELDS = [
    "interaction",
    "initial_state_energy",
    "initial_state_type",
    "initial_state_zenith",
    "initial_state_azimuth",
    "initial_state_x",
    "initial_state_y",
    "initial_state_z",
    "final_state_energy",
    "final_state_type",
    "bjorken_x",
    "bjorken_y",
    "column_depth",
]

EXPECTED_PHOTON_FIELDS = [
    "sensor_pos_x",
    "sensor_pos_y",
    "sensor_pos_z",
    "string_id",
    "sensor_id",
    "t",
    "id_idx",
]

N_EVENTS = 5

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def _pass(msg):
    print(f"  PASS: {msg}")

def _fail(msg):
    print(f"  FAIL: {msg}")

def run_smoketest():
    failures = []

    # --------------------------------------------------------
    # 1. Import checks
    # --------------------------------------------------------
    print("\n[1] Checking imports...")
    try:
        import prometheus  # noqa: F401
        _pass("import prometheus")
    except Exception as exc:
        _fail(f"import prometheus — {exc}")
        failures.append("import prometheus")

    try:
        import LeptonInjector  # noqa: F401
        _pass("import LeptonInjector")
    except Exception as exc:
        _fail(f"import LeptonInjector — {exc}")
        failures.append("import LeptonInjector")

    try:
        import LeptonWeighter  # noqa: F401
        _pass("import LeptonWeighter")
    except Exception as exc:
        _fail(f"import LeptonWeighter — {exc}")
        failures.append("import LeptonWeighter")

    if failures:
        print("\nFAIL: import errors prevent further testing.")
        return 1

    # --------------------------------------------------------
    # 2. Build config and run simulation
    # --------------------------------------------------------
    print("\n[2] Running simulation...")
    output_dir = tempfile.mkdtemp(prefix="prometheus_smoketest_")
    outfile = os.path.join(output_dir, "smoketest.parquet")

    try:
        from prometheus import Prometheus, config

        config["run"]["nevents"] = N_EVENTS
        config["run"]["random state seed"] = 42
        config["run"]["outfile"] = outfile
        config["run"]["storage prefix"] = output_dir

        config["detector"]["geo file"] = os.path.join(
            RESOURCE_DIR, "geofiles/demo_ice.geo"
        )

        config["injection"]["LeptonInjector"]["paths"]["xsec dir"] = os.path.join(
            RESOURCE_DIR, "cross_section_splines"
        )
        config["injection"]["LeptonInjector"]["simulation"]["minimal energy"] = 1e3
        config["injection"]["LeptonInjector"]["simulation"]["maximal energy"] = 1e4
        config["injection"]["LeptonInjector"]["simulation"]["final state 1"] = "MuMinus"
        config["injection"]["LeptonInjector"]["simulation"]["final state 2"] = "Hadrons"

        config["photon propagator"]["name"] = "PPC"
        config["photon propagator"]["PPC"]["paths"]["ppc_exe"] = os.path.join(
            RESOURCE_DIR, "PPC_executables/PPC/ppc"
        )
        config["photon propagator"]["PPC"]["paths"]["ppctables"] = os.path.join(
            RESOURCE_DIR, "PPC_tables/south_pole"
        )
        config["photon propagator"]["PPC"]["paths"]["ppc_tmpdir"] = os.path.join(
            output_dir, "ppc_tmp"
        )
        config["photon propagator"]["PPC"]["paths"]["force"] = True

        sim = Prometheus(userconfig=config)
        sim.sim()
        _pass("simulation completed without errors")
    except Exception as exc:
        _fail(f"simulation raised an exception — {exc}")
        traceback.print_exc()
        failures.append("simulation run")
        # Clean up and bail — remaining checks require the output file
        shutil.rmtree(output_dir, ignore_errors=True)
        print("\nFAIL: simulation did not complete; skipping output checks.")
        return 1

    # --------------------------------------------------------
    # 3. Verify output file exists and has expected structure
    # --------------------------------------------------------
    print("\n[3] Verifying output parquet...")
    try:
        import pyarrow.parquet as pq

        if not os.path.isfile(outfile):
            _fail(f"output file not found: {outfile}")
            failures.append("output file exists")
        else:
            _pass(f"output file exists: {outfile}")

            table = pq.read_table(outfile)
            top_fields = set(table.schema.names)

            # Check mc_truth is present
            if "mc_truth" in top_fields:
                _pass("top-level field present: mc_truth")
            else:
                _fail("top-level field missing: mc_truth")
                failures.append("field mc_truth")

            # photons is only written when at least one event produces a hit;
            # its absence is not itself a failure
            if "photons" in top_fields:
                _pass("top-level field present: photons")
            else:
                _pass("top-level field absent: photons (no events produced hits — OK)")

            # Check row count
            n_rows = table.num_rows
            if n_rows == N_EVENTS:
                _pass(f"row count matches: {n_rows} == {N_EVENTS}")
            else:
                _fail(f"row count mismatch: got {n_rows}, expected {N_EVENTS}")
                failures.append("row count")

            # Check mc_truth sub-fields using awkward-array (fields are
            # wrapped in AwkwardArrowType extension and not exposed via
            # pyarrow's .type.names)
            if "mc_truth" in top_fields:
                import awkward as ak
                mc_truth_arr = ak.from_arrow(table.column("mc_truth"))
                mc_truth_fields = set(ak.fields(mc_truth_arr))
                for field in EXPECTED_MC_TRUTH_FIELDS:
                    if field in mc_truth_fields:
                        _pass(f"mc_truth sub-field present: {field}")
                    else:
                        _fail(f"mc_truth sub-field missing: {field}")
                        failures.append(f"mc_truth.{field}")

            # Check photons sub-fields only when photons were produced
            if "photons" in top_fields:
                import awkward as ak
                photons_arr = ak.from_arrow(table.column("photons"))
                # photons is a struct-of-lists (one list per field, one entry per event)
                photon_fields = set(ak.fields(photons_arr))
                for field in EXPECTED_PHOTON_FIELDS:
                    if field in photon_fields:
                        _pass(f"photons sub-field present: {field}")
                    else:
                        _fail(f"photons sub-field missing: {field}")
                        failures.append(f"photons.{field}")

    except Exception as exc:
        _fail(f"output verification raised an exception — {exc}")
        traceback.print_exc()
        failures.append("output verification")
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    print()
    if failures:
        print(f"FAIL — {len(failures)} check(s) failed:")
        for f in failures:
            print(f"  - {f}")
        return 1
    else:
        print("PASS — all checks passed.")
        return 0


if __name__ == "__main__":
    sys.exit(run_smoketest())
