"""Run summary builder and emitter.

This module contains the logic to construct and emit the end-of-run
summary for a Prometheus run. It intentionally operates on a
``Prometheus`` instance passed in to avoid circular imports.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from time import time

from .config import config
from .utils.formatting import _file_checksum, _human_size


def emit_run_summary(prom, outfile, end_out, size=None):
    """Build and emit a run summary for the given Prometheus instance.

    Parameters
    ----------
    prom : Prometheus
        The Prometheus instance (used only to read attributes; not imported).
    outfile : str
        Path to the output file written by the run.
    end_out : float
        Timestamp when the output write finished (seconds since epoch).
    size : int or None
        Size in bytes of the output file if already known; if None the
        function will attempt to stat the file.
    """
    logger = logging.getLogger("prometheus.prometheus")

    try:
        try:
            events_written = len(prom.injection) if prom.injection is not None else 0
        except Exception:
            events_written = 0

        inj_time = float(
            getattr(prom, "_end_inj", float("nan")) - getattr(prom, "_start_inj", float("nan"))
        )
        prop_time = float(
            getattr(prom, "_end_prop", float("nan")) - getattr(prom, "_start_prop", float("nan"))
        )
        write_time = float(
            getattr(prom, "_end_out", end_out) - getattr(prom, "_start_out", float("nan"))
        )
        total_time = float(
            getattr(prom, "_end_out", end_out) - getattr(prom, "_run_start_time", float("nan"))
        )
        throughput = events_written / total_time if total_time and total_time > 0 else float("nan")

        start_ts = datetime.fromtimestamp(getattr(prom, "_run_start_time", time())).isoformat()
        end_ts = datetime.fromtimestamp(end_out).isoformat()

        logfile = getattr(config.run, "logfile", None) or "console"

        # External subprocess statuses (ppc)
        external_statuses = []
        try:
            from .photon_propagation import ppc_photon_propagator as _ppc_mod

            external_statuses = getattr(_ppc_mod, "subprocess_statuses", [])
        except Exception:
            external_statuses = []

        warn = err = crit = 0
        if getattr(prom, "_log_counter", None) is not None:
            warn = getattr(prom._log_counter, "warning_count", 0)
            err = getattr(prom._log_counter, "error_count", 0)
            crit = getattr(prom._log_counter, "critical_count", 0)
            getattr(prom._log_counter, "info_count", 0)

        # File size and checksum
        if size is None:
            try:
                size = Path(outfile).stat().st_size
            except Exception:
                size = None

        human = _human_size(size) if size is not None else "unknown"
        checksum = _file_checksum(outfile) if (size is not None and size < (1 << 30)) else ""

        mode = getattr(config.run, "summary_mode", "user") or "user"
        compact = bool(getattr(config.run, "compact", False))
        summary_json = bool(getattr(config.run, "summary_json", False))
        summary_json_path = getattr(config.run, "summary_json_path", None)

        success = bool(size is not None and size > 0 and Path(outfile).exists())

        # Compact single-line mode
        if compact:
            compact_line = (
                f"Run {getattr(config.run, 'run_number', 'unknown')} "
                f"{'✔' if success else '❌'} | {events_written} events"
                f" | {total_time:.2f} s | {throughput:.2f} ev/s"
                f" | output: {Path(str(outfile)).name if outfile else 'None'}"
            )
            if mode == "user":
                print(compact_line)
            else:
                logger.info(compact_line)
        else:
            # USER-friendly multi-line narrative
            header = "🚀 Prometheus Simulation"
            header_lines = []
            header_lines.append(header)
            header_lines.append("")
            header_lines.append(f"Run ID:        {getattr(config.run, 'run_number', 'unknown')}")
            header_lines.append(f"Events:        {getattr(config.run, 'nevents', 'unknown')}")
            header_lines.append(f"Injector:      {getattr(config.injection, 'name', 'unknown')}")
            header_lines.append(
                f"Propagator:    {getattr(config.photon_propagator, 'name', 'unknown')}"
            )
            modules = getattr(
                prom.detector, "n_modules", len(getattr(prom.detector, "modules", []))
            )
            header_lines.append(f"Modules:       {modules}")
            header_lines.append("")
            header_lines.append("─" * 64)

            # Phase 1 - Injection
            phase_lines = []
            phase_lines.append("")
            phase_lines.append("[1/3] Injection")
            if hasattr(prom, "_start_inj") and hasattr(prom, "_end_inj"):
                phase_lines.append(f"✔ Completed in {inj_time:.2f} s")
                phase_lines.append(f"  → Events generated: {events_written}")
            else:
                phase_lines.append("✖ Not completed")

            # Phase 2 - Propagation
            phase_lines.append("")
            phase_lines.append("[2/3] Propagation")
            if hasattr(prom, "_start_prop") and hasattr(prom, "_end_prop"):
                phase_lines.append(f"✔ Completed in {prop_time:.2f} s")
                try:
                    parts = set()
                    for ev in prom.injection:
                        for fs in getattr(ev, "final_states", []):
                            parts.add(str(fs))
                    parts_list = ", ".join(list(parts)[:5])
                    if parts_list:
                        phase_lines.append(f"  → Particles tracked: {parts_list}")
                except Exception:
                    pass
            else:
                phase_lines.append("✖ Not completed")

            # Phase 3 - Output
            phase_lines.append("")
            phase_lines.append("[3/3] Output")
            if success:
                phase_lines.append("✔ File written")
                phase_lines.append(f"  → {outfile} ({human})")
            else:
                phase_lines.append("✖ No output written")

            footer_lines = []
            footer_lines.append("")
            footer_lines.append("─" * 64)
            footer_lines.append("")
            footer_lines.append(
                "✅ Simulation completed successfully" if success else "❌ Simulation failed"
            )
            footer_lines.append("")
            footer_lines.append("Summary:")
            footer_lines.append(f"  Total time:     {total_time:.2f} s")
            footer_lines.append(f"  Throughput:     {throughput:.2f} events/s")
            footer_lines.append(f"  Output events:  {events_written}")
            footer_lines.append("")

            # Emit user view
            user_view = "\n".join(header_lines + phase_lines + footer_lines)
            if mode == "user":
                print(user_view)
            else:
                logger.info(user_view)

            # Prominent output pointer
            try:
                output_pointer = f"📦 Output ready:\n   {outfile}\n"
                if mode == "user":
                    print(output_pointer)
                else:
                    logger.info(output_pointer)
            except Exception:
                pass

            # Collate captured warnings and internal prints
            captured_warnings = getattr(prom, "_captured_warnings", []) or []
            init_noise = bool(
                getattr(prom, "_init_output", None) and getattr(prom, "_init_output", "").strip()
            )
            inject_noise = bool(
                getattr(prom, "_inject_output", None)
                and getattr(prom, "_inject_output", "").strip()
            )
            prop_noise = bool(
                getattr(prom, "_propagate_output", None)
                and getattr(prom, "_propagate_output", "").strip()
            )
            internal_noise_count = int(init_noise) + int(inject_noise) + int(prop_noise)
            total_warnings = len(captured_warnings) + (warn + err + crit)

            if total_warnings + internal_noise_count > 0:
                if mode == "user":
                    print(f"⚠ Warnings detected ({total_warnings + internal_noise_count})")
                    print(
                        "  Hint: set config.run.summary_mode='debug' "
                        "(and optionally config.run.verbosity='DEBUG') to inspect details"
                    )
                else:
                    logger.warning(
                        "⚠ Warnings detected (%d)", total_warnings + internal_noise_count
                    )
                    for w in captured_warnings:
                        logger.warning("  - %s", w)
                    if init_noise:
                        logger.debug("--- Init output (excerpt) ---")
                        for line_ in getattr(prom, "_init_output", "").splitlines()[:200]:
                            logger.debug(line_)
                    if inject_noise:
                        logger.debug("--- Injection output (excerpt) ---")
                        for line_ in getattr(prom, "_inject_output", "").splitlines()[:200]:
                            logger.debug(line_)
                    if prop_noise:
                        logger.debug("--- Propagation output (excerpt) ---")
                        for line_ in getattr(prom, "_propagate_output", "").splitlines()[:200]:
                            logger.debug(line_)

        # Debug/details mode: emit the richer, developer-oriented summary at DEBUG
        if mode.lower() == "debug":
            debug_lines = [
                (
                    f"Run {getattr(config.run, 'run_number', 'unknown')}"
                    f" | requested_nevents={getattr(config.run, 'nevents', 'unknown')}"
                    f" | events_written={events_written}"
                ),
                f"Timings [s]: inj={inj_time:.3f} prop={prop_time:.3f}"
                f" write={write_time:.3f} total={total_time:.3f}",
                f"Throughput: {throughput:.2f} ev/s",
                f"Start: {start_ts} | End: {end_ts}",
                f"Output: {outfile} ({human})",
                f"Checksum (sha256): {checksum}",
                f"Logs: {logfile}",
                f"External subprocess statuses: {external_statuses}",
                f"Warnings: {warn} Errors: {err} Critical: {crit}",
            ]
            logger.debug("Run debug summary:\n%s", "\n".join(debug_lines))

        # Optionally write a JSON summary for automation
        if summary_json and outfile:
            try:
                jpath = summary_json_path or (outfile + ".summary.json")
                jdata = {
                    "run_number": getattr(config.run, "run_number", None),
                    "requested_nevents": getattr(config.run, "nevents", None),
                    "events_written": events_written,
                    "timings": {
                        "inj": inj_time,
                        "prop": prop_time,
                        "write": write_time,
                        "total": total_time,
                    },
                    "throughput_ev_s": throughput,
                    "output": {"path": outfile, "size": size, "checksum": checksum},
                    "logs": logfile,
                    "external_statuses": external_statuses,
                    "warnings": {"warning": warn, "error": err, "critical": crit},
                    "start": start_ts,
                    "end": end_ts,
                    "success": success,
                }
                with open(jpath, "w") as jf:
                    json.dump(jdata, jf, indent=2)
                logger.info("Wrote run JSON summary to %s", jpath)
            except Exception:
                logger.debug("Failed to write JSON summary to %s", jpath)

    except Exception:
        logger.debug("Failed to produce run summary")


__all__ = ["emit_run_summary"]
