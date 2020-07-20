"""Unility functions for manipulatiing nifti (.nii) files"""

import SimpleITK as sitk
import numpy as np
from matplotlib import pylab as plt


def load_image_array_from_nii(nii_path):
    sitk_img = sitk.ReadImage(nii_path)
    image_array_3d = sitk.GetArrayFromImage(sitk_img)
    return image_array_3d


def load_sitk_image(nii_path):
    sitk_img = sitk.ReadImage(nii_path)
    return sitk_img


def write_array_to_nii(image_array_3d, header_itk_img=None, output_path=None):
    """Write array to itk image with optional header

    Args:
        image_array_3d: image array to write array to
        header_itk_img: optional, itk image to copy header from
        output_path:

    Returns:

    """
    img_aligned = sitk.GetImageFromArray(image_array_3d)
    if header_itk_img is not None:
        try:
            try:
                img_aligned.CopyInformation(header_itk_img)
            except:
                img_aligned.SetDirection(header_itk_img.GetDirection())
                img_aligned.SetOrigin(header_itk_img.GetOrigin())
                img_aligned.SetSpacing(header_itk_img.GetSpacing())
        except:
            # for 2d nii image, there is no such header info
            print('Cannot copy header from {}'.format(header_itk_img))
    if output_path:
        print('Writing to {}'.format(output_path))
        sitk.WriteImage(img_aligned, output_path)
    return img_aligned


def resample_3d_by_spacing(seg_img, new_spacing=(1, 1, 1)):
    """Resample seg_img with new_spacing

    Args:
        seg_img:
        new_spacing:

    Returns:

    """
    old_spacing = seg_img.GetSpacing()
    new_spacing = new_spacing
    new_size = tuple(np.array(seg_img.GetSize()) * np.array(old_spacing) / np.array(new_spacing))
    new_size = [int(x) for x in new_size]
    resampled_seg_img = sitk.Resample(seg_img, new_size, sitk.Transform(),
                                      sitk.sitkNearestNeighbor, seg_img.GetOrigin(),
                                      new_spacing, seg_img.GetDirection(), 0.0,
                                      seg_img.GetPixelIDValue())
    return resampled_seg_img


def myshow(img, spacing=None, title=None, figsize=(6, 6), fig=None):
    """

    Args:
        img: sitk image or numpy array
        spacing:
        title:

    Returns:

    """
    if isinstance(img, np.ndarray):
        nda = img
    else:
        nda = sitk.GetArrayViewFromImage(img)

    if spacing is None:
        if isinstance(img, np.ndarray):
            spacing = (1, 1)
        else:
            spacing = img.GetSpacing()

    ysize, xsize = nda.shape[:2]
    if fig is None:
        fig = plt.figure(title, figsize=figsize)
    extent = (0, xsize * spacing[1], ysize * spacing[0], 0)

    t = plt.imshow(nda,
                   extent=extent,
                   interpolation='hamming',
                   cmap='gray',
                   origin='upper')
    # plt.colorbar()

    if (title):
        plt.title(title)

    return fig
