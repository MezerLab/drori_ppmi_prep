import argparse

from drori_ppmi_prep.segmentation.synthseg import run_synthseg


def main():
    parser = argparse.ArgumentParser(
        description="Run FreeSurfer SynthSeg for one input image."
    )
    parser.add_argument("--input", required=True, help="Input image.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--synthseg-cmd", default="mri_synthseg")
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()
    output_file, status = run_synthseg(
        input_image=args.input,
        output_dir=args.output_dir,
        synthseg_cmd=args.synthseg_cmd,
        overwrite=args.overwrite,
    )

    print(f"SynthSeg status: {status}")
    print(f"Output dir     : {args.output_dir}")
    if output_file is not None:
        print(f"Segmentation   : {output_file}")

    return 0 if status in {"done", "skipped"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
