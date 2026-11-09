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
    """Create an oracle baseline by downsampling the CT image rasterizing the mesh onto it then upsampling the mask to the original resolution using trilinear interpolation."""
    if isinstance(ct_image, str):
        ct_image=sitk.ReadImage(ct_image)
    elif not isinstance(ct_image, sitk.Image):
        raise ValueError("ct_image must be a SimpleITK Image or a path to an image file.")
    
    new_size = _calculate_new_size(ct_image.GetSize(), ct_image.GetSpacing(), new_spacing)

    downsampled_reference=sitk.Image(new_size, sitk.sitkUInt8)
    downsampled_reference.SetSpacing(new_spacing)
    downsampled_reference.SetOrigin(ct_image.GetOrigin())
    downsampled_reference.SetDirection(ct_image.GetDirection())
    
    mask_image=stl_to_ct_mask(stl_path, ct_image=downsampled_reference)
    upsampled_mask = trilinear_interpolation(mask_image, ct_image.GetSpacing())
    
    return upsampled_mask

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
        0.0,
        sitk.sitkFloat32
    )
    
    upsampled_mask = sitk.GetArrayFromImage(upsampled) > threshold
    upsampled_mask = sitk.GetImageFromArray(upsampled_mask.astype(np.uint8))
    upsampled_mask.CopyInformation(upsampled)
    
    return upsampled_mask

def probability_mask_interpolation(image:sitk.Image, prob_map:sitk.Image, new_spacing, threshold=0.5):
    """Upsample the probability map to the new spacing using trilinear interpolation."""

    new_size = _calculate_new_size(image.GetSize(), image.GetSpacing(), new_spacing)

    upsampled = sitk.Resample(
        prob_map,
        new_size,
        sitk.Transform(),
        sitk.sitkLinear,
        prob_map.GetOrigin(),
        new_spacing,
        prob_map.GetDirection(),
        0.0,
        sitk.sitkFloat32
    )
    
    upsampled_mask = sitk.GetArrayFromImage(upsampled) > threshold
    upsampled_mask = sitk.GetImageFromArray(upsampled_mask.astype(np.uint8))
    upsampled_mask.CopyInformation(upsampled)

    return upsampled_mask

def sdf_interpolation(mask_image:sitk.Image, new_spacing):
    """Upsample the mask image to the new spacing using signed distance field interpolation."""

    new_size = _calculate_new_size(mask_image.GetSize(), mask_image.GetSpacing(), new_spacing)
    
    sdf_in= sitk.SignedMaurerDistanceMap(mask_image, insideIsPositive=False, squaredDistance=False, useImageSpacing=True)
    
    inv = 1-mask_image

    sdf_out = sitk.SignedMaurerDistanceMap(inv, insideIsPositive=False,squaredDistance=False, useImageSpacing=True)
    sdf_image = 0.5 * (sdf_in - sdf_out)

    upsampled = sitk.Resample(
        sdf_image,
        new_size,
        sitk.Transform(),
        sitk.sitkLinear,
        sdf_image.GetOrigin(),
        new_spacing,
        sdf_image.GetDirection(),
        0.0,
        sitk.sitkFloat32
    )
    
    upsampled_mask = sitk.GetArrayFromImage(upsampled) < 0
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
    
    temp_image = sitk.Image(dims.tolist(), sitk.sitkUInt8)
    temp_image.SetSpacing(spacing)
    temp_image.SetOrigin(origin)
    img=stl_to_ct_mask(stl_path, ct_image=temp_image)

    metrics = calculate_metrics(img, stl_path, stl_is_ras=False)
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
    
    results = []
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
        
        targets=["femur_left","femur_right"]
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
            ts_prob_map=sitk.ReadImage(os.path.join(ts_path,target+"_prob_1p5mm.nii.gz"))
            
        except Exception as e:
            raise ValueError(f"Error reading TotalSegmentator predictions for {subject} in {ts_path}: {e}")
        

        subject_id=int(subject[3:])
        
        
        #TotalSegmentator low resolution metrics
        ts_metrics = calculate_metrics(ts_low_res_mask_image, stl_path)
        ts_metrics["baseline"] = "TotalSegmentator"
        ts_metrics["subject_id"] = subject_id
        results.append(ts_metrics)

        #TotalSegmentator full resolution metrics
        ts_full_res_metrics = calculate_metrics(ts_full_res_mask_image, stl_path)
        ts_full_res_metrics["baseline"] = "TotalSegmentator_nearest_neighbor"
        ts_full_res_metrics["subject_id"] = subject_id
        results.append(ts_full_res_metrics)

        # Oracle baseline
        oracle_mask_image = oracle(ct_image, stl_path, ts_spacing)
        oracle_metrics = calculate_metrics(oracle_mask_image, stl_path)
        oracle_metrics["baseline"] = "oracle"
        oracle_metrics["subject_id"] = subject_id
        results.append(oracle_metrics)
        
        # Round trip baseline
        round_trip_metrics = round_trip(stl_path, ts_spacing, margin=round_trip_margin) # Round trip error should be larger on lower resolution, so this is an upper bound
        round_trip_metrics["baseline"] = "round_trip"
        round_trip_metrics["subject_id"] = subject_id
        results.append(round_trip_metrics)

        # Trilinear interpolation baseline
        trilinear_mask = trilinear_interpolation(ts_low_res_mask_image, ct_image.GetSpacing(), threshold=threshold)
        trilinear_metrics = calculate_metrics(trilinear_mask, stl_path)
        trilinear_metrics["baseline"] = "trilinear_interpolation"
        trilinear_metrics["subject_id"] = subject_id
        results.append(trilinear_metrics)
        
        # Probability mask interpolation baseline
        prob_mask = probability_mask_interpolation(ts_low_res_image, ts_prob_map, ts_spacing, threshold=threshold)
        prob_metrics = calculate_metrics(prob_mask, stl_path)
        prob_metrics["baseline"] = "probability_mask_interpolation"
        prob_metrics["subject_id"] = subject_id
        results.append(prob_metrics)

        # SDF interpolation baseline
        sdf_mask = sdf_interpolation(ts_low_res_mask_image, ct_image.GetSpacing())
        sdf_metrics = calculate_metrics(sdf_mask, stl_path)
        sdf_metrics["baseline"] = "sdf_interpolation"
        sdf_metrics["subject_id"] = subject_id
        results.append(sdf_metrics)

    return results