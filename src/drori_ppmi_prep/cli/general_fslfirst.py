import argparse
from pathlib import Path

from drori_ppmi_prep.segmentation.first import run_fsl_first
from drori_ppmi_prep.segmentation.utils import erode_label_segmentation


def main():
    parser = argparse.ArgumentParser(
        description="Run FSL FIRST for one input image."
    )
    parser.add_argument("--input", required=True, help="Input T1/reference image.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--first-cmd", default="run_first_all")
    parser.add_argument("--brain-extracted", action="store_true")
    parser.add_argument("--erode", action="store_true", help="Also write eroded segmentation.")
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    _, status = run_fsl_first(
        input_image=args.input,
        output_dir=output_dir,
        first_cmd=args.first_cmd,
        overwrite=args.overwrite,
        brain_extracted=args.brain_extracted,
    )

    print(f"FSL FIRST status: {status}")
    print(f"Output dir      : {output_dir}")

    if status in {"done", "skipped"} and args.erode:
        eroded = erode_label_segmentation(
            segmentation_file=output_dir / "first_all_fast_firstseg.nii.gz",
            output_file=output_dir / "first_all_fast_firstseg_eroded.nii.gz",
            iterations=1,
            overwrite=args.overwrite,
        )
        if eroded is None:
            print("Erosion status  : failed")
            return 1
        print(f"Eroded output   : {eroded}")

    return 0 if status in {"done", "skipped"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
