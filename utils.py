import warnings

import numpy as np
import SimpleITK as sitk
import trimesh, vtk, os, tqdm
import pyvista as pv
import nibabel as nib
from skimage import measure
from pathlib import Path

from totalsegmentator.python_api import totalsegmentator
from totalsegmentator.alignment import undo_canonical
from totalsegmentator.map_to_binary import class_map, class_map_5_parts, map_taskid_to_partname_ct
import totalsegmentator.nnunet as ts_nnunet

from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk, numpy_to_vtkIdTypeArray
from surface_distance import metrics as surface_metrics


from config import HFValid, pred_path

def stl_to_ct_mask(stl_path, ct_path=None, ct_image=None, output_path=None):
    
    if ct_image is None and ct_path is None:
        raise ValueError("Either ct_path or ct_image must be provided.")
    
    mesh=trimesh.load_mesh(stl_path, file_type='stl')
    if mesh.is_empty:
        raise ValueError(f"Mesh is empty: {stl_path}")
    
    if not mesh.is_watertight:
        raise ValueError(
            "The STL mesh is not watertight. "
        )
    
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    
    if ct_image is None:
        ct_image=sitk.ReadImage(ct_path)
    
    size=np.asarray(ct_image.GetSize())
    spacing=np.asarray(ct_image.GetSpacing())
    origin=np.asarray(ct_image.GetOrigin())
    direction=np.asarray(ct_image.GetDirection()).reshape(3, 3)
    
    #conversion matrix and index coordinates
    physical_to_index_matrix = np.linalg.inv(direction) @ np.diag(1 / spacing)
    
    index_vertices = (vertices - origin) @ physical_to_index_matrix.T
    
    if(index_vertices.min() < 0 or index_vertices.max() >= size).any():
        warnings.warn(
            "Some vertices are outside the CT image bounds. "
            "The resulting mask may be incomplete."
        )
    
    #vtk points and faces
    vtk_points = vtk.vtkPoints()
    vtk_points.SetNumberOfPoints(len(index_vertices))
    vtk_points.SetData(numpy_to_vtk(index_vertices, deep=True))
    
    n_faces=len(mesh.faces)
    offsets=np.arange(0, n_faces * 3+1, 3, dtype=np.int64)
    connectivity=np.asarray(mesh.faces, dtype=np.int64).flatten()
    
    vtk_faces=vtk.vtkCellArray()
    vtk_faces.SetData(
        numpy_to_vtkIdTypeArray(offsets, deep=True),
        numpy_to_vtkIdTypeArray(connectivity, deep=True)
    )
    vtk_mesh=vtk.vtkPolyData()
    vtk_mesh.SetPoints(vtk_points)
    vtk_mesh.SetPolys(vtk_faces)
    
    # Create a unit-spaced grid in index coordinates.
    reference = vtk.vtkImageData()
    reference.SetDimensions(*[int(v) for v in size])
    reference.SetOrigin(0, 0, 0)
    reference.SetSpacing(1, 1, 1)
    reference.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
    
    vtk_to_numpy(
        reference.GetPointData().GetScalars()
    ).fill(1)
    
    poly_to_stencil = vtk.vtkPolyDataToImageStencil()
    poly_to_stencil.SetInputData(vtk_mesh)
    poly_to_stencil.SetOutputOrigin(0, 0, 0)
    poly_to_stencil.SetOutputSpacing(1, 1, 1)
    poly_to_stencil.SetOutputWholeExtent(reference.GetExtent())
    poly_to_stencil.Update()
    
    stencil = vtk.vtkImageStencil()
    stencil.SetInputData(reference)
    stencil.SetStencilConnection(
        poly_to_stencil.GetOutputPort()
    )
    stencil.ReverseStencilOff()
    stencil.SetBackgroundValue(0)
    stencil.Update()
    
    result = vtk_to_numpy(
        stencil.GetOutput().GetPointData().GetScalars()
    ).reshape(size[2], size[1], size[0])

    return result.astype(bool)


def image_to_mesh(img: sitk.Image, level=0.5):
    """Marching cubes on a SimpleITK image -> trimesh in physical (LPS, mm) coordinates."""
    vol = sitk.GetArrayFromImage(img).astype(np.float32)   # (z, y, x)
    pad_value = vol.min() if level == 0 else 0.0           # SDF: pad with "outside" (most negative or positive depends on your sign convention, see note)
    vol = np.pad(vol, 1, constant_values=pad_value)
    verts_zyx, faces, _, _ = measure.marching_cubes(vol, level=level)
    idx_xyz = verts_zyx[:, ::-1] - 1.0                     # undo padding, reorder to (x, y, z)

    spacing = np.array(img.GetSpacing())
    origin = np.array(img.GetOrigin())
    D = np.array(img.GetDirection()).reshape(3, 3)
    verts_mm = origin + (idx_xyz * spacing) @ D.T
    return trimesh.Trimesh(verts_mm, faces, process=False)


def _closest_distances(mesh_pv: pv.PolyData, pts: np.ndarray) -> np.ndarray:
    # exact point-to-triangle distances
    _, closest = mesh_pv.find_closest_cell(pts, return_closest_point=True)
    return np.linalg.norm(pts - closest, axis=1)


def calculate_metrics(
    pred: sitk.Image,          # binary mask, probability map, or signed distance
    gt_stl_path,
    stl_is_ras=False,          # True if the STL is in RAS (3D Slicer export)
    level=0.5,
    n_points=100_000,
    tolerances_mm=(0.5, 1.0),
    seed=0,
):
    if not (sitk.GetArrayViewFromImage(pred) > level).any():
        raise ValueError("Prediction has no foreground.")

    gt_mesh = trimesh.load_mesh(gt_stl_path, force="mesh")
    if stl_is_ras:
        gt_mesh.vertices[:, :2] *= -1                      # RAS -> LPS

    pred_mesh = image_to_mesh(pred, level=level)

    offset = np.linalg.norm(gt_mesh.bounds.mean(axis=0) - pred_mesh.bounds.mean(axis=0))
    if offset > 30:
        warnings.warn(f"GT and prediction are {offset:.0f} mm apart; check RAS/LPS and origin.")

    pts_gt, _ = trimesh.sample.sample_surface(gt_mesh, n_points, seed=seed)
    pts_pred, _ = trimesh.sample.sample_surface(pred_mesh, n_points, seed=seed)

    gt_pv = pv.wrap(gt_mesh)       # trimesh -> pyvista
    pred_pv = pv.wrap(pred_mesh)
    d_gt_to_pred = _closest_distances(pred_pv, pts_gt)
    d_pred_to_gt = _closest_distances(gt_pv, pts_pred)

    assd = 0.5 * (d_gt_to_pred.mean() + d_pred_to_gt.mean())
    hd95 = max(np.percentile(d_gt_to_pred, 95), np.percentile(d_pred_to_gt, 95))

    a_gt, a_pred = gt_mesh.area, pred_mesh.area
    surface_dice = {
        tau: float((a_gt * np.mean(d_gt_to_pred <= tau) + a_pred * np.mean(d_pred_to_gt <= tau))
                   / (a_gt + a_pred))
        for tau in tolerances_mm
    }
    return {"assd_mm": float(assd), "hd95_mm": float(hd95), "surface_dice": surface_dice}


TARGETS = ["femur_left", "femur_right"]   # extend for other bones

_total_inv = {n: i for i, n in class_map["total"].items()}        # name -> global label
_where = {n: (part, idx) for part, cmap in class_map_5_parts.items()
          for idx, n in cmap.items()}                              # name -> (sub-model, channel)
_tid_of_part = {v: k for k, v in map_taskid_to_partname_ct.items()}


def _to_original_orientation(data, affine, img_orig):
    """Wrap `data` (TotalSegmentator's canonical orientation) as NIfTI and undo the reorientation."""
    return undo_canonical(nib.Nifti1Image(data, affine), img_orig)


def run_total_with_probs(nifti_path, out_prefix, device, targets, work_dir):
    """Standard full `total` run (1.5 mm model, no roi_subset) that also captures its 1.5 mm data.

    Two internals of `totalsegmentator.nnunet` are hooked for the duration of the call:
      * nnUNetv2_predict: probabilities are saved only for the sub-model(s) containing `targets`
        (with a single path every sub-model would overwrite the previous one's file).
      * change_spacing:   captures the CT as resampled to 1.5 mm and the merged 1.5 mm label map
                          just before TotalSegmentator upsamples it to the original grid.

    The full-resolution segmentation is written to `out_prefix` as usual.

    Returns {"image": nib image, "labels": nib image, "probs": {target: nib image}}, all on the
    1.5 mm grid, in the orientation of the input, each carrying its own affine.
    `work_dir` only holds temporary probability files, which are removed again.
    """
    work_dir = Path(work_dir)
    wanted = {}   # task id -> [(name, channel)]
    for t in targets:
        part, ch = _where[t]
        wanted.setdefault(_tid_of_part[part], []).append((t, ch))
    prob_files = {tid: work_dir / f"probs_{tid}.npz" for tid in wanted}

    captured = {}
    orig_predict, orig_resample = ts_nnunet.nnUNetv2_predict, ts_nnunet.change_spacing

    def predict_hook(dir_in, dir_out, task_id, *args, save_probabilities_path=None, **kwargs):
        return orig_predict(dir_in, dir_out, task_id, *args,
                            save_probabilities_path=prob_files.get(task_id), **kwargs)

    def resample_hook(img_in, new_spacing=1.25, target_shape=None, *args, **kwargs):
        result = orig_resample(img_in, new_spacing, target_shape, *args, **kwargs)
        if target_shape is None:                 # forward call: CT -> 1.5 mm
            captured.setdefault("image", result)
        else:                                    # back call: merged 1.5 mm labels -> original grid
            # copy the data now; img_in may be backed by a temp file that TotalSegmentator deletes
            captured["labels"] = np.asarray(img_in.get_fdata()).astype(np.uint8)
        return result

    ts_nnunet.nnUNetv2_predict, ts_nnunet.change_spacing = predict_hook, resample_hook
    try:
        totalsegmentator(input=nifti_path, output=out_prefix, ml=True, device=device, quiet=True)
    finally:
        ts_nnunet.nnUNetv2_predict, ts_nnunet.change_spacing = orig_predict, orig_resample

    if "image" not in captured or "labels" not in captured:
        raise RuntimeError("Could not capture TotalSegmentator's 1.5 mm data; the hooks target "
                           "`totalsegmentator.nnunet.change_spacing`, check the installed version.")

    img_orig = nib.load(nifti_path)
    image, labels_xyz = captured["image"], captured["labels"]
    if labels_xyz.shape != image.shape:
        raise ValueError(f"Label map {labels_xyz.shape} does not match 1.5 mm image {image.shape}.")

    result = {
        "image": undo_canonical(image, img_orig),
        "labels": _to_original_orientation(labels_xyz, image.affine, img_orig),
        "probs": {},
    }
    for tid, items in wanted.items():
        try:
            with np.load(prob_files[tid]) as npz:
                probs = npz["probabilities"]                       # (channels, z, y, x)
            for name, ch in items:
                prob_xyz = np.ascontiguousarray(probs[ch].transpose(2, 1, 0), dtype=np.float32)
                if prob_xyz.shape != image.shape:
                    raise ValueError(f"{name}: probabilities {prob_xyz.shape} do not match "
                                     f"1.5 mm image {image.shape}.")
                result["probs"][name] = _to_original_orientation(prob_xyz, image.affine, img_orig)
            del probs
        finally:
            prob_files[tid].unlink(missing_ok=True)
            prob_files[tid].with_suffix(".pkl").unlink(missing_ok=True)
    return result


def ts_predict(subject_ids: list[int] | str, device: str = "gpu", output_dir: str = pred_path):
    """Run TotalSegmentator (standard 1.5 mm `total` model) on HFValid subjects.

    Written to <output_dir>/<subject>/:
        ml_segmentation.nii            full-res multilabel segmentation (TotalSegmentator output)
        image_1p5mm.nii.gz             CT as resampled for the model
        ml_segmentation_1p5mm.nii.gz   merged 1.5 mm label map, before upsampling
      and for each femur found:
        <femur>.nii                    full-res binary mask
        <femur>_mask_1p5mm.nii.gz      1.5 mm binary mask
        <femur>_prob_1p5mm.nii.gz      1.5 mm probability map
    All 1.5 mm files are NIfTI in the input's orientation, so SimpleITK aligns them by geometry.
    """
    subjects_dir = os.path.join(HFValid, "Subjects")
    if isinstance(subject_ids, str):
        if subject_ids.lower() != "all":
            raise ValueError(f"Invalid subject_ids string: {subject_ids}. Use 'all' or a list of integers.")
        available_subjects = os.listdir(subjects_dir)
        if len(available_subjects) != 101:
            raise ValueError("Missing patients. Check the data.")
    else:
        subjects = {f"Pat{i:03d}" for i in subject_ids}
        available_subjects = subjects & set(os.listdir(subjects_dir))
        if not available_subjects:
            raise ValueError(f"No patients found in {subjects_dir}. Check the path.")
        if available_subjects != subjects:
            raise ValueError(f"Missing patients: {subjects - available_subjects}")

    femur_labels = {name: _total_inv[name] for name in TARGETS}

    for subject in tqdm.tqdm(sorted(available_subjects), desc="TotalSegmentator predictions"):
        subject_dir = os.path.join(subjects_dir, subject)
        out_dir = os.path.join(output_dir, subject)
        os.makedirs(out_dir, exist_ok=True)

        # Convert NRRD to NIfTI
        image = sitk.ReadImage(os.path.join(subject_dir, f"{subject}.nrrd"))
        nifti_path = os.path.join(subject_dir, f"{subject}.nii.gz")
        sitk.WriteImage(image, nifti_path)

        lowres = run_total_with_probs(nifti_path, os.path.join(out_dir, "ml_segmentation"),
                                      device, TARGETS, out_dir)
        nib.save(lowres["image"], os.path.join(out_dir, "image_1p5mm.nii.gz"))
        nib.save(lowres["labels"], os.path.join(out_dir, "ml_segmentation_1p5mm.nii.gz"))

        segmentation = sitk.ReadImage(os.path.join(out_dir, "ml_segmentation.nii"))
        seg_array = sitk.GetArrayFromImage(segmentation)
        lowres_array = np.asanyarray(lowres["labels"].dataobj)

        if (~np.isin(seg_array, [0, *femur_labels.values()])).any():
            warnings.warn(f"Other labels are present for {subject}.")

        found = [name for name, label in femur_labels.items() if (seg_array == label).any()]
        if not found:
            warnings.warn(f"No femur found for {subject}.")
        elif len(found) > 1:
            warnings.warn(f"Both femur left and right masks are present for {subject}. Check the segmentation.")

        for name in found:
            label = femur_labels[name]

            mask = sitk.GetImageFromArray((seg_array == label).astype(np.uint8))
            mask.CopyInformation(segmentation)
            sitk.WriteImage(mask, os.path.join(out_dir, f"{name}.nii"))

            lowres_mask = nib.Nifti1Image((lowres_array == label).astype(np.uint8), lowres["labels"].affine)
            nib.save(lowres_mask, os.path.join(out_dir, f"{name}_mask_1p5mm.nii.gz"))
            nib.save(lowres["probs"][name], os.path.join(out_dir, f"{name}_prob_1p5mm.nii.gz"))


def vertex_lattice_check(stl_path, nrrd_path, stl_is_ras=False, eps=0.03):
    img = sitk.ReadImage(nrrd_path)
    v = np.asarray(trimesh.load_mesh(stl_path).vertices, dtype=float).copy()
    if stl_is_ras:
        v[:, :2] *= -1  # RAS -> LPS

    spacing = np.array(img.GetSpacing())
    origin = np.array(img.GetOrigin())
    D = np.array(img.GetDirection()).reshape(3, 3)

    # physical -> continuous index (x, y, z); integer index = voxel centre
    idx = ((v - origin) @ D) / spacing

    # sanity: are the vertices inside the volume at all?
    size = np.array(img.GetSize())
    inside = np.all((idx > -0.5) & (idx < size - 0.5), axis=1).mean()

    frac = idx % 1.0
    on_centre = (frac < eps) | (frac > 1 - eps)   # fractional part ~ 0
    on_mid = np.abs(frac - 0.5) < eps             # fractional part ~ 0.5
    on_lattice = on_centre | on_mid

    all3 = on_lattice.all(axis=1).mean()
    one_mid = ((on_mid.sum(axis=1) == 1) & (on_centre.sum(axis=1) == 2)).mean()
    chance = (4 * eps) ** 3                       # expected if fractional parts were uniform

    print(f"vertices inside volume:                {inside:.3f}")
    print(f"all 3 axes on {{0, 0.5}} lattice:         {all3:.3f}  (chance ~ {chance:.4f})")
    print(f"exactly one axis at 0.5, two at 0:     {one_mid:.3f}")
    for ax, name in enumerate("xyz"):
        h, _ = np.histogram(frac[:, ax], bins=10, range=(0, 1))
        print(f"axis {name} fractional-part histogram:", (h / h.sum()).round(3)) 