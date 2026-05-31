import argparse
import json

from drori_ppmi_prep.analysis.mrgrad import (
    DEFAULT_PRESETS,
    list_mrgrad_presets,
    load_mrgrad_config,
    load_mrgrad_preset,
    run_mrgrad_analysis,
)


def print_result(name, output, status):
    if status == "done":
        print(f"DONE: {name}: {output}")
    elif status == "skipped":
        print(f"SKIPPED: already done: {name}: {output}")
    elif status == "missing_command":
        print(f"SKIPPED: MATLAB command not found: {name}")
    elif status == "missing":
        print(f"SKIPPED: missing required inputs or mrGrad toolbox: {name}")
    else:
        raise RuntimeError(f"mrGrad analysis failed: {name}. Check the output logs.")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run mrGrad v2.0.3 group analyses from built-in presets or "
            "custom JSON configuration files."
        )
    )
    parser.add_argument("output_root", nargs="?")
    parser.add_argument("--preset", action="append", default=[])
    parser.add_argument("--config", action="append", default=[])
    parser.add_argument("--list-presets", action="store_true")
    parser.add_argument("--show-preset", default=None)
    parser.add_argument("--mrgrad-dir", default=None)
    parser.add_argument("--mrgrad-no-download", action="store_true")
    parser.add_argument("--matlab-cmd", default="matlab")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    if args.list_presets:
        print("\n".join(list_mrgrad_presets()))
        return

    if args.show_preset is not None:
        print(json.dumps(load_mrgrad_preset(args.show_preset), indent=2))
        return

    if args.output_root is None:
        parser.error("output_root is required unless --list-presets or --show-preset is used")

    selected_presets = args.preset
    if not selected_presets and not args.config:
        selected_presets = DEFAULT_PRESETS

    analyses = [
        (name, load_mrgrad_preset(name))
        for name in selected_presets
    ]
    analyses.extend(
        (config["analysis_name"], config)
        for config in (load_mrgrad_config(path) for path in args.config)
    )

    for name, config in analyses:
        output, status = run_mrgrad_analysis(
            output_root=args.output_root,
            config=config,
            mrgrad_dir=args.mrgrad_dir,
            matlab_cmd=args.matlab_cmd,
            allow_download=not args.mrgrad_no_download,
            overwrite=args.overwrite,
            parallel=args.parallel,
        )
        print_result(name, output, status)


if __name__ == "__main__":
    main()
