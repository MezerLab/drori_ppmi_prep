from pathlib import Path
from scipy.ndimage import binary_erosion

import nibabel as nib
import numpy as np


def matlab_sphere_1_structure():
    structure = np.zeros((3, 3, 3), dtype=bool)
    structure[1, 1, 1] = True
    structure[0, 1, 1] = True
    structure[2, 1, 1] = True
    structure[1, 0, 1] = True
    structure[1, 2, 1] = True
    structure[1, 1, 0] = True
    structure[1, 1, 2] = True
    return structure


def erode_label_segmentation(
    segmentation_file,
    output_file=None,
    iterations=1,
    overwrite=False,
):
    segmentation_file = Path(segmentation_file)

    if output_file is None:
        output_file = segmentation_file.with_name(
            segmentation_file.name.replace(".nii.gz", "_eroded.nii.gz")
        )

    output_file = Path(output_file)

    if output_file.exists() and not overwrite:
        return output_file

    if not segmentation_file.exists():
        return None

    img = nib.load(str(segmentation_file))
    data = img.get_fdata()

    labels = np.unique(data)
    labels = labels[labels != 0]

    eroded_data = np.zeros_like(data, dtype=np.int16)

    # Match MATLAB strel("sphere", 1): center voxel plus 6 face neighbors.
    structure = matlab_sphere_1_structure()

    for label in labels:
        mask = data == label

        eroded_mask = binary_erosion(
            mask,
            structure=structure,
            iterations=iterations,
        )

        eroded_data[eroded_mask] = int(label)

    out_img = nib.Nifti1Image(
        eroded_data,
        affine=img.affine,
        header=img.header,
    )

    out_img.set_data_dtype(np.int16)
    nib.save(out_img, str(output_file))

    return output_file
