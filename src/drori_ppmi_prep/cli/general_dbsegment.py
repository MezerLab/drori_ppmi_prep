import argparse

from drori_ppmi_prep.segmentation.dbsegment import run_dbsegment


def main():
    parser = argparse.ArgumentParser(
        description="Run DBSegment for one input image."
    )
    parser.add_argument("--input", required=True, help="Input T1/reference image.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--dbsegment-cmd", default="DBSegment")
    parser.add_argument("--cpu", action="store_true", help="Disable CUDA for DBSegment.")
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()
    output_file, status = run_dbsegment(
        input_image=args.input,
        output_dir=args.output_dir,
        model_path=args.model_path,
        dbsegment_cmd=args.dbsegment_cmd,
        overwrite=args.overwrite,
        use_cuda=not args.cpu,
    )

    print(f"DBSegment status: {status}")
    print(f"Output dir      : {args.output_dir}")
    if output_file is not None:
        print(f"Segmentation    : {output_file}")

    return 0 if status in {"done", "skipped"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
