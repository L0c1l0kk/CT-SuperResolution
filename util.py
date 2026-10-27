import warnings

import numpy as np
import SimpleITK as sitk
import trimesh, vtk

from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk, numpy_to_vtkIdTypeArray
from surface_distance import metrics as surface_metrics



HFValid="home/loci/data/HFValid_Collection_v3"
metadata=HFValid+"/HFValid_Metadata.csv"

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


def calculate_metrics(pred_mask, gt_mask, spacing):
    if not gt_mask.any() or not pred_mask.any():
        raise ValueError("Both masks must contain foreground voxels.")
    
    spacing_zyx = tuple(reversed(spacing))

    surface_distances = surface_metrics.compute_surface_distances(
        mask_gt=gt_mask,
        mask_pred=pred_mask,
        spacing_mm=spacing_zyx,
    )

    # Mean distance in each direction:
    # ground truth -> prediction, prediction -> ground truth
    directed_asd = surface_metrics.compute_average_surface_distance(
        surface_distances
    )
    assd_mm = float(np.mean(directed_asd))
    hd95_mm = float(
        surface_metrics.compute_robust_hausdorff(
            surface_distances,
            percent=95,
        )
    )
    
    dice = float(
        surface_metrics.compute_dice_coefficient(
            gt_mask,
            pred_mask,
        )
    )
    
    return (assd_mm, hd95_mm, dice)