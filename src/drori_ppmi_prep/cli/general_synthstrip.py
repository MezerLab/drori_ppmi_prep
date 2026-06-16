import argparse

from drori_ppmi_prep.preprocessing.synthstrip import run_synthstrip


def main():
    parser = argparse.ArgumentParser(
        description="Run SynthStrip for one input image."
    )
    parser.add_argument("--input", required=True, help="Input NIfTI image.")
    parser.add_argument("--output", required=True, help="Output brain-extracted image.")
    parser.add_argument("--mask", default=None, help="Optional output brain mask.")
    parser.add_argument("--synthstrip-cmd", default="mri_synthstrip")
    parser.add_argument("--no-csf", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()
    run_synthstrip(
        input_nii=args.input,
        output_nii=args.output,
        mask_nii=args.mask,
        synthstrip_cmd=args.synthstrip_cmd,
        overwrite=args.overwrite,
        no_csf=args.no_csf,
    )

    print(f"SynthStrip output: {args.output}")
    if args.mask is not None:
        print(f"SynthStrip mask  : {args.mask}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
