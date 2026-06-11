from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


IMAGE_PATHS = {
    "t1_space": {
        "t1": "t1_space/T1.nii.gz",
        "t2": "t1_space/T2.nii.gz",
        "pd": "t1_space/PD.nii.gz",
    },
    "mri_unbias_deg2": {
        "t1": "t1_space/mri_unbias_deg2/T1.nii.gz",
        "t2": "t1_space/mri_unbias_deg2/T2.nii.gz",
        "pd": "t1_space/mri_unbias_deg2/PD.nii.gz",
    },
}
INTENSITY_STATS = ("median", "mean", "mad", "std")
ROI_STATS = INTENSITY_STATS + ("volume",)

ASEG_LABELS = {
    2: "Left-Cerebral-White-Matter",
    3: "Left-Cerebral-Cortex",
    4: "Left-Lateral-Ventricle",
    5: "Left-Inf-Lat-Vent",
    7: "Left-Cerebellum-White-Matter",
    8: "Left-Cerebellum-Cortex",
    10: "Left-Thalamus",
    11: "Left-Caudate",
    12: "Left-Putamen",
    13: "Left-Pallidum",
    14: "3rd-Ventricle",
    15: "4th-Ventricle",
    16: "Brain-Stem",
    17: "Left-Hippocampus",
    18: "Left-Amygdala",
    24: "CSF",
    26: "Left-Accumbens-area",
    28: "Left-VentralDC",
    41: "Right-Cerebral-White-Matter",
    42: "Right-Cerebral-Cortex",
    43: "Right-Lateral-Ventricle",
    44: "Right-Inf-Lat-Vent",
    46: "Right-Cerebellum-White-Matter",
    47: "Right-Cerebellum-Cortex",
    49: "Right-Thalamus",
    50: "Right-Caudate",
    51: "Right-Putamen",
    52: "Right-Pallidum",
    53: "Right-Hippocampus",
    54: "Right-Amygdala",
    58: "Right-Accumbens-area",
    60: "Right-VentralDC",
}
SYNTHSEG_LABELS = dict(ASEG_LABELS)
APARC_NAMES = (
    "unknown",
    "bankssts",
    "caudalanteriorcingulate",
    "caudalmiddlefrontal",
    "corpuscallosum",
    "cuneus",
    "entorhinal",
    "fusiform",
    "inferiorparietal",
    "inferiortemporal",
    "isthmuscingulate",
    "lateraloccipital",
    "lateralorbitofrontal",
    "lingual",
    "medialorbitofrontal",
    "middletemporal",
    "parahippocampal",
    "paracentral",
    "parsopercularis",
    "parsorbitalis",
    "parstriangularis",
    "pericalcarine",
    "postcentral",
    "posteriorcingulate",
    "precentral",
    "precuneus",
    "rostralanteriorcingulate",
    "rostralmiddlefrontal",
    "superiorfrontal",
    "superiorparietal",
    "superiortemporal",
    "supramarginal",
    "frontalpole",
    "temporalpole",
    "transversetemporal",
    "insula",
)
DKT_APARC_INDICES = tuple(
    index
    for index in range(len(APARC_NAMES))
    if index not in {0, 1, 4, 32, 33}
)
DKT_APARC_ASEG_LABELS = {
    **ASEG_LABELS,
    30: "Left-vessel",
    31: "Left-choroid-plexus",
    62: "Right-vessel",
    63: "Right-choroid-plexus",
    72: "5th-Ventricle",
    77: "WM-hypointensities",
    80: "non-WM-hypointensities",
    85: "Optic-Chiasm",
    251: "CC_Posterior",
    252: "CC_Mid_Posterior",
    253: "CC_Central",
    254: "CC_Mid_Anterior",
    255: "CC_Anterior",
    **{1000 + index: f"ctx-lh-{APARC_NAMES[index]}" for index in DKT_APARC_INDICES},
    **{2000 + index: f"ctx-rh-{APARC_NAMES[index]}" for index in DKT_APARC_INDICES},
}
FIRST_LABELS = {
    key: ASEG_LABELS[key]
    for key in (10, 11, 12, 13, 16, 17, 18, 26, 49, 50, 51, 52, 53, 54, 58)
}
DBSEGMENT_LABELS = {
    1: "Brain-mask",
    2: "Caudate-L",
    3: "Caudate-R",
    4: "GPe-L",
    5: "GPe-R",
    6: "GPi-L",
    7: "GPi-R",
    8: "Habenular-nuclei-L",
    9: "Habenular-nuclei-R",
    10: "Internal-capsule-L",
    11: "Internal-capsule-R",
    12: "Nucleus-accumbens-L",
    13: "Nucleus-accumbens-R",
    14: "Putamen-L",
    15: "Putamen-R",
    16: "Red-nucleus-L",
    17: "Red-nucleus-R",
    18: "SNc-L",
    19: "SNc-R",
    20: "SNr-L",
    21: "SNr-R",
    22: "STN-L",
    23: "STN-R",
    24: "Thalamus-L",
    25: "Thalamus-R",
    26: "VPL-L",
    27: "VPL-R",
    28: "Lateral-ventricle-L",
    29: "Lateral-ventricle-R",
    30: "VIM-L",
    31: "VIM-R",
}
DBSEGMENT_GP_SN_LABELS = {
    4: "GP-L",
    5: "GP-R",
    18: "SN-L",
    19: "SN-R",
}
MASSP_LABELS = {
    index + 1: label_name
    for index, label_name in enumerate(
        (
            "Str-L",
            "Str-R",
            "STN-L",
            "STN-R",
            "SN-L",
            "SN-R",
            "RN-L",
            "RN-R",
            "GPi-L",
            "GPi-R",
            "GPe-L",
            "GPe-R",
            "Tha-L",
            "Tha-R",
            "LV-L",
            "LV-R",
            "3V",
            "4V",
            "Amg-L",
            "Amg-R",
            "ic-L",
            "ic-R",
            "VTA-L",
            "VTA-R",
            "fx",
            "PAG-L",
            "PAG-R",
            "PPN-L",
            "PPN-R",
            "Cl-L",
            "Cl-R",
        )
    )
}

SEGMENTATIONS = {
    "freesurfer": {
        "path": (
            "t1_space/segmentation/freesurfer/t1_space_outputs/"
            "aparc.DKTatlas+aseg.nii.gz"
        ),
        "labels": "freesurfer",
    },
    "synthstrip": {
        "path": "t1_space/segmentation/synthstrip/T1_brainmask_mask.nii.gz",
        "labels": {1: "Brain-mask"},
    },
    "synthseg": {
        "path": "t1_space/segmentation/synthseg/synthseg.nii.gz",
        "labels": SYNTHSEG_LABELS,
    },
    "fslfirst": {
        "path": "t1_space/segmentation/fslfirst/first_all_fast_firstseg.nii.gz",
        "labels": FIRST_LABELS,
    },
    "fslfirst_eroded": {
        "path": "t1_space/segmentation/fslfirst/first_all_fast_firstseg_eroded.nii.gz",
        "labels": FIRST_LABELS,
    },
    "dbsegment": {
        "path": "t1_space/segmentation/dbsegment/T1.nii.gz",
        "labels": DBSEGMENT_LABELS,
    },
    "dbsegment_GP_SN": {
        "path": "t1_space/segmentation/dbsegment/derivatives/GP_SN_seg.nii.gz",
        "labels": DBSEGMENT_GP_SN_LABELS,
    },
    "massp": {
        "path": (
            "t1_space/segmentation/massp/ahead2sub_ants/"
            "massp2021-parcellation_decade-61to80_2ref.nii.gz"
        ),
        "labels": MASSP_LABELS,
    },
}


def resolve_dataset_paths(output_root):
    output_root = Path(output_root)
    config_path = output_root / "ppmi_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = json.loads(config_path.read_text())
    return Path(config["analysis_root"]), Path(config["metadata_csv"])


def parse_freesurfer_lut(path):
    labels = {}
    with Path(path).open() as f:
        for line in f:
            fields = line.split()
            if len(fields) >= 2 and fields[0].isdigit():
                labels[int(fields[0])] = fields[1]
    return labels


def resolve_freesurfer_lut(path=None):
    candidates = []
    if path is not None:
        candidates.append(Path(path))
    freesurfer_home = os.environ.get("FREESURFER_HOME")
    if freesurfer_home:
        candidates.append(Path(freesurfer_home) / "FreeSurferColorLUT.txt")

    for candidate in candidates:
        if candidate.exists():
            lut = parse_freesurfer_lut(candidate)
            return {
                label: lut.get(label, fallback_name)
                for label, fallback_name in DKT_APARC_ASEG_LABELS.items()
            }

    return dict(DKT_APARC_ASEG_LABELS)


def _load_image(path):
    try:
        image = nib.load(str(path))
        data = np.asanyarray(image.dataobj)
        if data.ndim != 3:
            return None, None
        return image, data
    except Exception:
        return None, None


def _intensity_values(values, stats):
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {stat: np.nan for stat in stats}

    result = {}
    median = None
    if "median" in stats or "mad" in stats:
        median = float(np.median(values))
    if "median" in stats:
        result["median"] = median
    if "mean" in stats:
        result["mean"] = float(np.mean(values))
    if "mad" in stats:
        result["mad"] = float(np.median(np.abs(values - median)))
    if "std" in stats:
        result["std"] = float(np.std(values))
    return result


def _session_stats(job):
    (
        row_index,
        analysis_root,
        subject_id,
        session_id,
        segmentations,
        image_paths,
        intensity_stats,
        include_volume,
    ) = job
    session_dir = Path(analysis_root) / subject_id / session_id
    result = {"row_index": row_index, "stats": {}}
    image_cache = {}

    def load_image(image_set, image_name, relative_path):
        key = (image_set, image_name)
        if key not in image_cache:
            image_cache[key] = _load_image(session_dir / relative_path)
        return image_cache[key]

    for segmentation_name, config in segmentations.items():
        segmentation_image, segmentation_data = _load_image(session_dir / config["path"])
        labels = config["labels"]
        segmentation_result = {"volume": {}, "images": {}}
        result["stats"][segmentation_name] = segmentation_result

        if segmentation_data is None:
            continue

        segmentation_data = np.rint(segmentation_data).astype(np.int32)
        voxel_volume = float(abs(np.linalg.det(segmentation_image.affine[:3, :3])))
        compatible_images = {}
        if intensity_stats:
            for image_set, paths in image_paths.items():
                compatible_images[image_set] = {}
                for image_name, relative_path in paths.items():
                    image, image_data = load_image(image_set, image_name, relative_path)
                    if (
                        image_data is None
                        or image_data.shape != segmentation_data.shape
                        or not np.allclose(image.affine, segmentation_image.affine)
                    ):
                        continue
                    compatible_images[image_set][image_name] = image_data
                    segmentation_result["images"].setdefault(image_set, {})[image_name] = {}

        for label in labels:
            mask = segmentation_data == label
            if include_volume:
                segmentation_result["volume"][label] = (
                    float(np.count_nonzero(mask)) * voxel_volume
                )
            if not intensity_stats:
                continue
            for image_set, image_data_by_name in compatible_images.items():
                for image_name, image_data in image_data_by_name.items():
                    segmentation_result["images"][image_set][image_name][label] = (
                        _intensity_values(image_data[mask], intensity_stats)
                    )

    return result


def _metadata_rows(metadata_csv):
    metadata = pd.read_csv(metadata_csv, dtype={"SubjectID": str, "SessionID": str})
    required = {"SubjectID", "SessionID"}
    if not required <= set(metadata.columns):
        raise ValueError("Metadata CSV must contain SubjectID and SessionID columns.")
    return metadata[["SubjectID", "SessionID"]].copy()


def _normalize_subject_id(value):
    value = str(value).strip()
    try:
        number = float(value)
        if number.is_integer():
            return str(int(number))
    except ValueError:
        pass
    return value


def _select_image_paths(selected_images):
    if selected_images is None:
        return {
            image_set: dict(paths)
            for image_set, paths in IMAGE_PATHS.items()
        }

    selected_images = list(dict.fromkeys(selected_images))
    valid_images = {
        image_name
        for paths in IMAGE_PATHS.values()
        for image_name in paths
    }
    unknown = sorted(set(selected_images) - valid_images)
    if unknown:
        available = ", ".join(sorted(valid_images))
        raise ValueError(
            f"Unknown image(s): {', '.join(unknown)}. "
            f"Available images: {available}"
        )

    return {
        image_set: {
            image_name: relative_path
            for image_name, relative_path in paths.items()
            if image_name in selected_images
        }
        for image_set, paths in IMAGE_PATHS.items()
    }


def _select_stats(selected_stats):
    if selected_stats is None:
        selected_stats = ROI_STATS
    selected_stats = list(dict.fromkeys(selected_stats))
    unknown = sorted(set(selected_stats) - set(ROI_STATS))
    if unknown:
        available = ", ".join(ROI_STATS)
        raise ValueError(
            f"Unknown stat(s): {', '.join(unknown)}. "
            f"Available stats: {available}"
        )
    return (
        tuple(stat for stat in selected_stats if stat in INTENSITY_STATS),
        "volume" in selected_stats,
    )


def _output_files(output_root, segmentations, image_paths, intensity_stats, include_volume):
    output_root = Path(output_root)
    files = []
    for segmentation_name in segmentations:
        if include_volume:
            files.append(
                output_root
                / "group_analysis"
                / "ROI_stats"
                / "t1_space"
                / f"{segmentation_name}_volume.csv"
            )
        for image_set in image_paths:
            output_dir = output_root / "group_analysis" / "ROI_stats" / "t1_space"
            if image_set != "t1_space":
                output_dir = output_dir / image_set
            for image_name in image_paths[image_set]:
                for stat in intensity_stats:
                    files.append(output_dir / f"{segmentation_name}_{image_name}_{stat}.csv")
    return files


def _roi_column_name(label_name):
    return label_name.replace("-", "_")


def _progress_message(completed, total, interval):
    if interval <= 0:
        return False
    return completed == total or completed == 1 or completed % interval == 0


def run_roi_stats(
    output_root,
    freesurfer_lut=None,
    overwrite=False,
    parallel=False,
    max_workers=None,
    selected_segmentations=None,
    selected_images=None,
    selected_stats=None,
    progress=True,
    progress_interval=50,
):
    output_root = Path(output_root)
    analysis_root, metadata_csv = resolve_dataset_paths(output_root)
    metadata = _metadata_rows(metadata_csv)
    segmentations = {
        name: dict(config)
        for name, config in SEGMENTATIONS.items()
    }
    if selected_segmentations is not None:
        selected_segmentations = list(dict.fromkeys(selected_segmentations))
        unknown = sorted(set(selected_segmentations) - set(segmentations))
        if unknown:
            available = ", ".join(segmentations)
            raise ValueError(
                f"Unknown segmentation(s): {', '.join(unknown)}. "
                f"Available segmentations: {available}"
            )
        segmentations = {
            name: segmentations[name]
            for name in selected_segmentations
        }

    if "freesurfer" in segmentations:
        segmentations["freesurfer"]["labels"] = resolve_freesurfer_lut(freesurfer_lut)

    image_paths = _select_image_paths(selected_images)
    intensity_stats, include_volume = _select_stats(selected_stats)

    output_files = _output_files(
        output_root,
        segmentations,
        image_paths,
        intensity_stats,
        include_volume,
    )
    if not overwrite and all(path.exists() for path in output_files):
        return output_root / "group_analysis" / "ROI_stats", "skipped"

    jobs = [
        (
            row_index,
            str(analysis_root),
            _normalize_subject_id(row["SubjectID"]),
            str(row["SessionID"]),
            segmentations,
            image_paths,
            intensity_stats,
            include_volume,
        )
        for row_index, row in metadata.iterrows()
    ]
    total_jobs = len(jobs)
    if progress:
        segmentation_names = ", ".join(segmentations)
        image_names = ", ".join(
            dict.fromkeys(
                image_name
                for paths in image_paths.values()
                for image_name in paths
            )
        )
        stat_names = ", ".join((*intensity_stats, *(("volume",) if include_volume else ())))
        print(
            "Running ROI stats: "
            f"{total_jobs} sessions; "
            f"segmentations={segmentation_names}; "
            f"images={image_names or 'none'}; "
            f"stats={stat_names or 'none'}",
            flush=True,
        )

    if parallel:
        session_results = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_session_stats, job) for job in jobs]
            for completed, future in enumerate(as_completed(futures), start=1):
                session_results.append(future.result())
                if progress and _progress_message(completed, total_jobs, progress_interval):
                    print(f"  processed {completed}/{total_jobs} sessions", flush=True)
    else:
        session_results = []
        for completed, job in enumerate(jobs, start=1):
            session_results.append(_session_stats(job))
            if progress and _progress_message(completed, total_jobs, progress_interval):
                print(f"  processed {completed}/{total_jobs} sessions", flush=True)

    session_results = {result["row_index"]: result["stats"] for result in session_results}
    base_columns = metadata.reset_index(drop=True)

    def write_table(path, segmentation_name, labels, value_getter):
        roi_columns = {
            _roi_column_name(label_name): [
                value_getter(session_results[row_index].get(segmentation_name, {}), label)
                for row_index in metadata.index
            ]
            for label, label_name in labels.items()
        }
        table = pd.concat(
            [base_columns, pd.DataFrame(roi_columns, index=base_columns.index)],
            axis=1,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False)

    if progress:
        print("Writing ROI stats tables...", flush=True)

    for segmentation_name, config in segmentations.items():
        labels = config["labels"]
        output_dir = output_root / "group_analysis" / "ROI_stats" / "t1_space"
        if include_volume:
            write_table(
                output_dir / f"{segmentation_name}_volume.csv",
                segmentation_name,
                labels,
                lambda result, label: result.get("volume", {}).get(label, np.nan),
            )
        for image_set in image_paths:
            image_output_dir = output_dir if image_set == "t1_space" else output_dir / image_set
            for image_name in image_paths[image_set]:
                for stat in intensity_stats:
                    write_table(
                        image_output_dir / f"{segmentation_name}_{image_name}_{stat}.csv",
                        segmentation_name,
                        labels,
                        lambda result, label, image_set=image_set, image_name=image_name, stat=stat: (
                            result.get("images", {})
                            .get(image_set, {})
                            .get(image_name, {})
                            .get(label, {})
                            .get(stat, np.nan)
                        ),
                    )

    return output_root / "group_analysis" / "ROI_stats", "done"
