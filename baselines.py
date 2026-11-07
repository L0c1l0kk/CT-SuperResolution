import warnings

import numpy as np
import SimpleITK as sitk
import trimesh, os, tqdm
import pyvista as pv

from utils import stl_to_ct_mask, image_to_mesh, calculate_metrics
from config import HFValid, pred_path

def _calculate_new_size(old_size, old_spacing, new_spacing):
    """Calculate the new size of an image given the old size, old spacing, and new spacing."""
    return [
        round(n * old_spacing / new_spacing)
        for n, old_spacing, new_spacing in zip(old_size, old_spacing, new_spacing)
    ]

def oracle(ct_image:sitk.Image, stl_path, new_spacing):
    """Create an oracle baseline by downsampling the CT image and creating a mask from the STL file.
    Returns a tuple of the downsampled CT image and the mask."""
    if isinstance(ct_image, str):
        ct_image=sitk.ReadImage(ct_image)
    elif not isinstance(ct_image, sitk.Image):
        raise ValueError("ct_image must be a SimpleITK Image or a path to an image file.")
    
    new_size = _calculate_new_size(ct_image.GetSize(), ct_image.GetSpacing(), new_spacing)

    downsampled_image = sitk.Resample(
        ct_image,
        new_size,
        sitk.Transform(),
        sitk.sitkLinear,
        ct_image.GetOrigin(),
        new_spacing,
        ct_image.GetDirection(),
        0,
        ct_image.GetPixelID()
    )
    
    mask=stl_to_ct_mask(stl_path, ct_image=downsampled_image)
    mask_image = sitk.GetImageFromArray(mask.astype(np.uint8))
    mask_image.CopyInformation(downsampled_image)
    return (downsampled_image, mask_image)

def trilinear_interpolation(mask_image:sitk.Image, new_spacing, threshold=0.5):
    """Upsample the mask image to the new spacing using trilinear interpolation."""

    new_size = _calculate_new_size(mask_image.GetSize(), mask_image.GetSpacing(), new_spacing)

    upsampled = sitk.Resample(
        mask_image,
        new_size,
        sitk.Transform(),
        sitk.sitkLinear,
        mask_image.GetOrigin(),
        new_spacing,
        mask_image.GetDirection(),
        0,
        mask_image.GetPixelID()
    )
    
    upsampled_mask = sitk.GetArrayFromImage(upsampled) > threshold
    upsampled_mask = sitk.GetImageFromArray(upsampled_mask.astype(np.uint8))
    upsampled_mask.CopyInformation(upsampled)
    
    return upsampled_mask

def probability_mask_interpolation(image:sitk.Image, prob_map:np.ndarray, new_spacing, threshold=0.5):
    """Upsample the probability map to the new spacing using trilinear interpolation."""

    new_size = _calculate_new_size(image.GetSize(), image.GetSpacing(), new_spacing)

    prob_map_image = sitk.GetImageFromArray(prob_map.astype(np.float32))
    prob_map_image.CopyInformation(image)

    upsampled = sitk.Resample(
        prob_map_image,
        new_size,
        sitk.Transform(),
        sitk.sitkLinear,
        prob_map_image.GetOrigin(),
        new_spacing,
        prob_map_image.GetDirection(),
        0,
        prob_map_image.GetPixelID()
    )
    
    upsampled_mask = sitk.GetArrayFromImage(upsampled) > threshold
    upsampled_mask = sitk.GetImageFromArray(upsampled_mask.astype(np.uint8))
    upsampled_mask.CopyInformation(upsampled)

    return upsampled_mask


# to measure evaluation error
def round_trip(stl_path, spacing:np.ndarray, margin=5.0):
    """Convert an STL file to a CT mask and back to a mesh, returning the evaluation metrics. 
    This measures the error introduced by the conversion to a mesh in calcualte_metrics"""
    
    spacing=np.asarray(spacing)
    mesh = pv.read(stl_path)
    b = np.asarray(mesh.bounds).reshape(3, 2)
    origin = b[:, 0] - margin
    dims = np.ceil((b[:, 1] - b[:, 0] + 2 * margin) / spacing).astype(int) + 1

    # axis-aligned grid in the STL's own frame, so orientation can't interfere
    grid = pv.ImageData(dimensions=dims, spacing=spacing, origin=tuple(origin))
    sel = grid.select_enclosed_points(mesh, tolerance=0.0, check_surface=False)
    inside = np.asarray(sel["SelectedPoints"]).reshape(dims, order="F")   # (x, y, z)

    img = sitk.GetImageFromArray(inside.transpose(2, 1, 0).astype(np.uint8))  # (z, y, x)
    img.SetSpacing(spacing)
    img.SetOrigin(tuple(origin))

    metrics = calculate_metrics(img, stl_path, stl_is_ras=False)
    tm = trimesh.load_mesh(stl_path)
    metrics["volume_ratio_mask_to_stl"] = inside.sum() * np.prod(spacing) / tm.volume  # needs a watertight STL
    return metrics

def run_all_baselines(subject_ids:list[int]|str,data_path=HFValid, round_trip_margin=5.0, threshold=0.5, ts_pred_path=pred_path):
    
    """Run all baselines for a list of subject IDs or a single subject ID. 
    Returns a dictionary of metrics for each baseline and each subject."""
    if isinstance(subject_ids, str):
        if subject_ids.lower() == "all":
            available_subjects = os.listdir(HFValid+"/Subjects")
            if(len(available_subjects) != 101):
                raise ValueError(f"Missing patients. Check the data.")
        else:
            raise ValueError(f"Invalid subject_ids string: {subject_ids}. Use 'all' or a list of integers.")
    else:
        patients=set([f"Pat{str(i).zfill(3)}" for i in subject_ids])
        available_subjects=patients.intersection(set(os.listdir(HFValid+"/Subjects")))
        if len(available_subjects)!=len(patients):
            if len(available_subjects)==0:
                raise ValueError(f"No patients found in {HFValid}/Subjects. Check the path.")
            missing_patients=patients.difference(available_subjects)
            raise ValueError(f"Missing patients: {missing_patients}")
    
    results = {}
    for subject in tqdm.tqdm(sorted(available_subjects), desc="Running baselines"):
        
        stl_path = os.path.join(data_path, "Subjects", subject, f"{subject}.stl")
        ct_path = os.path.join(data_path, "Subjects", subject, f"{subject}.nrrd")
        if not os.path.isfile(stl_path):
            raise ValueError(f"STL file not found for {subject} at {stl_path}.")
        if not os.path.isfile(ct_path):
            raise ValueError(f"CT file not found for {subject} at {ct_path}.")
        ct_image = sitk.ReadImage(ct_path)
        
        ts_path = os.path.join(ts_pred_path, subject)
        if not os.path.exists(ts_path):
            raise ValueError(f"TotalSegmentator predictions not found for {subject} in {ts_pred_path}. Run TotalSegmentator first.")
        
        targets=["femur_left,femur_right"]
        target=""
        if all(element in os.listdir(ts_path) for element in [target+".nii" for target in targets]):
            warnings.warn(f"TotalSegmentator predicted both femurs for {subject}, skipping subject.")
            continue
        elif not any(element in os.listdir(ts_path) for element in ["femur_left.nii","femur_right.nii",]):
            raise ValueError(f"segmentation.nii not found for {subject} in {ts_path}.")
        else:
            if "femur_left.nii" in os.listdir(ts_path):
                target=targets[0]
            elif "femur_right.nii" in os.listdir(ts_path):
                target=targets[1]
        ts_spacing = np.array([1.5, 1.5, 1.5])  # TotalSegmentator spacing        
        
        try:
            ts_low_res_image=sitk.ReadImage(os.path.join(ts_path, "image_1p5mm.nii.gz"))
            ts_full_res_mask_image=sitk.ReadImage(os.path.join(ts_path, target+".nii"))
            ts_low_res_mask_image=sitk.ReadImage(os.path.join(ts_path,target+"_mask_1p5mm.nii.gz"))
            ts_prob_map=np.load(os.path.join(ts_path,target+"_prob_1p5mm.nii.gz"))
            
        except Exception as e:
            raise ValueError(f"Error reading TotalSegmentator predictions for {subject} in {ts_path}: {e}")
        

        subject_id=int(subject[3:])

        #TotalSegmentator metrics
        ts_metrics = calculate_metrics(ts_full_res_mask_image, stl_path)
        ts_metrics["baseline"] = "TotalSegmentator"
        ts_metrics["subject_id"] = subject_id

        # Oracle baseline
        _, oracle_mask_image = oracle(ct_image, stl_path, ts_spacing)
        oracle_metrics = calculate_metrics(oracle_mask_image, stl_path)
        oracle_metrics["baseline"] = "oracle"
        oracle_metrics["subject_id"] = subject_id
        
        # Trilinear interpolation baseline
        trilinear_mask = trilinear_interpolation(ts_low_res_mask_image, ct_image.GetSpacing(), threshold=threshold)
        trilinear_metrics = calculate_metrics(trilinear_mask, stl_path)
        trilinear_metrics["baseline"] = "trilinear_interpolation"
        trilinear_metrics["subject_id"] = subject_id
        
        # Probability mask interpolation baseline
        prob_mask = probability_mask_interpolation(ts_low_res_image, ts_prob_map, ts_spacing, threshold=threshold)
        prob_metrics = calculate_metrics(prob_mask, stl_path)
        prob_metrics["baseline"] = "probability_mask_interpolation"
        prob_metrics["subject_id"] = subject_id

        results[subject] = {
            "oracle": oracle_metrics,
            "trilinear_interpolation": trilinear_metrics,
            "probability_mask_interpolation": prob_metrics
        }
    
    return results