from skimage.feature import blob_log
import numpy as np
from fishspot.filter import white_tophat
import cv2
from PIL import Image, ImageDraw


def delete_point(round1_spots, round2_spots, delete_threshold=50):
    """
    Some pairs of points in the feature point set are too different and need to be eliminated

    parameter
    ---------
    delete_threshold: Maximum value of the difference between the horizontal or vertical coordinates of the matched points
    """
    diff = cv2.absdiff(round1_spots, round2_spots)
    delete_index = []
    for i in range(diff.shape[0]):
        for j in range(diff.shape[1]):
            if diff[i, j] > delete_threshold:
                delete_index.append(i)
                break
    round1_spots = np.delete(round1_spots, delete_index, axis=0)
    round2_spots = np.delete(round2_spots, delete_index, axis=0)
    return round1_spots, round2_spots


def get_feature_spots(image, min_sigma=5, max_sigma=12, threshold=0.1, exclude_border=50):
    """
    Applying LoG filters to find feature points

    """
    num_sigma = max_sigma - min_sigma
    spots = blob_log(image, min_sigma, max_sigma, num_sigma, threshold=threshold, exclude_border=exclude_border)
    return spots


def get_contexts(image, coords, radius):
    """
    Get neighborhoods of a set of coordinates

    Parameters
    ----------
    image : nd-array
        The source image data
    coords : nd-array Nx2
        A set of coordinates into the image data
    radius : scalar int
        The half width of neighborhoods to extract

    Returns
    -------
    neighborhoods : list of nd-arrays
        List of the extracted neighborhoods
    """

    contexts = []
    coords = coords.astype(int)
    for coord in coords:
        crop = tuple(slice(x-radius, x+radius+1) for x in coord)
        contexts.append(image[crop])
    return contexts


def pairwise_correlation(A, B):
    """
    Calculate the difference value between each of the two descriptors by directly subtracting the image pixel value

    Parameters
    ----------
    A : list of nd-arrays
        First list of neighborhoods
    B : list of nd-arrays
        Second list of neighborhoods

    Returns
    -------
    correlations : 2d-array, NxM
        N is the length of A and M is the length of B
    """
    corr = np.zeros((len(A), len(B)))
    i = 0
    j = 0
    for a in A:
        for b in B:
            # 计算图像差异
            corr[i, j] = np.sum(cv2.absdiff(a, b))
            j += 1
        j = 0
        i += 1
    return corr


def match_points(a_pos, b_pos, scores, threshold):
    """
    Given two point sets and pairwise scores, determine which points correspond.

    Parameters
    ----------
    a_pos : 2d-array Nx3
        First set of point coordinates
    b_pos : 2d-array Mx3
        Second set of point coordinates
    scores : 2d-array NxM
        Correspondence scores for all points in a_pos to all points in b_pos
    threshold : scalar float
        Minimum correspondence score for a valid match

    Returns
    -------
    matched_a_points, matched_b_points : two 2d-arrays both Px3
        The points from a_pos and b_pos that correspond
    """

    # get lowest scores above threshold
    best_indcs = np.argmin(scores, axis=1)
    a_indcs = range(len(a_pos))
    keeps = scores[(a_indcs, best_indcs)] < threshold

    # return positions of corresponding points
    return a_pos[keeps, :2], b_pos[best_indcs[keeps], :2]


def Registration(fix, mov, fix_feature_point_threshold=0.1, fix_feature_spot_min_sigma=4, fix_feature_spot_max_sigma=6, mov_feature_point_threshold=0.04, mov_feature_spot_min_sigma=2, mov_feature_spot_max_sigma=5, match_threshold=60000000):
    """
    Alignment using a feature point-based approach

    parameter
    ---------
    fix: 2d-array
        fixed image data
    mov: 2d-array
        moving image data
    feature_point_threshold: Used to regulate the number of feature points, the higher the threshold, the fewer the number of feature points
    feature_spot_min_sigma: Minimum radius of the bright spot around the feature
    feature_spot_max_sigma: Maximum radius of the bright spot around the feature
    match_threshold: The maximum value of the difference between the absolute values of the pixel values around the two clusters of point clouds, which can control the degree of matching

    return
    --------
    fix_point_t: 2d-array NX2
        The coordinates of the feature points found in the fixed graph, one feature point per row
    mov_spots_t: 2d-array NX2
        The coordinates of the feature points found in the moving graph, one feature point per row
    affine: 2d-array 2x3
        affine transformation matrix
    affined_mov: Shift matrix after affine transformation and Cubic interpolation

    """

    # -----------------------------------------------pre processing------------------------------------------------- #
    new_fix = white_tophat(fix, 5)
    new_mov = white_tophat(mov, 5)

    # ------------------------------------------find feature spots--------------------------------------------- #
    print('computing fix spots', flush=True)
    fix_spots = get_feature_spots(new_fix, threshold=fix_feature_point_threshold, min_sigma=fix_feature_spot_min_sigma, max_sigma=fix_feature_spot_max_sigma)
    print(f'found {len(fix_spots)} fix spots')

    print('computing mov spots', flush=True)
    mov_spots = get_feature_spots(new_mov, threshold=mov_feature_point_threshold, min_sigma=mov_feature_spot_min_sigma, max_sigma=mov_feature_spot_max_sigma)
    print(f'found {len(mov_spots)} mov spots')

    # -------------------------------------------------sort---------------------------------------------------- #
    nspots = 5000
    print('sorting spots', flush=True)
    sort_idx = np.argsort(fix_spots[:, 2])[::-1]
    fix_spots = fix_spots[sort_idx, :2][:nspots]
    sort_idx = np.argsort(mov_spots[:, 2])[::-1]
    mov_spots = mov_spots[sort_idx, :2][:nspots]

    # ----------------------------------------------get contexts------------------------------------------------ #
    cc_radius = 50
    print('extracting contexts', flush=True)
    fix_spot_contexts = get_contexts(fix, fix_spots, cc_radius)
    mov_spot_contexts = get_contexts(mov, mov_spots, cc_radius)

    # ---------------------------------------get point correspondences------------------------------------------- #
    print('computing pairwise correlations', flush=True)
    correlations = pairwise_correlation(fix_spot_contexts, mov_spot_contexts)
    fix_spots, mov_spots = match_points(fix_spots, mov_spots, correlations, match_threshold)
    # diff = cv2.absdiff(fix_spots, mov_spots)  # for observation

    # 在diff中可以看出有很多点配准不太正确，需要对其进行删除
    delete_threshold = 50  # 对应点中坐标差值在50以上的点剔除
    fix_spots, mov_spots = delete_point(fix_spots, mov_spots, delete_threshold)
    print(f'{len(fix_spots)} matched fix spots')
    print(f'{len(mov_spots)} matched mov spots')
    # diff_new = cv2.absdiff(fix_spots, mov_spots)  # for observation
    # diff_new = fix_spots - mov_spots  # for observation

    # 需要将spots的横纵坐标交换
    fix_spots_t = np.zeros((fix_spots.shape[0], fix_spots.shape[1]))
    fix_spots_t[:, 0] = fix_spots[:, 1]
    fix_spots_t[:, 1] = fix_spots[:, 0]
    mov_spots_t = np.zeros((mov_spots.shape[0], mov_spots.shape[1]))
    mov_spots_t[:, 0] = mov_spots[:, 1]
    mov_spots_t[:, 1] = mov_spots[:, 0]

    # ------------------------------------------------align--------------------------------------------------- #
    print('aligning', flush=True)
    align_threshold = 0.7
    affine, m = cv2.estimateAffine2D(mov_spots_t, fix_spots_t, ransacReprojThreshold=align_threshold)

    # -----------------------------------------------resample------------------------------------------------- #
    print('transforming', flush=True)
    output_size = (fix.shape[1], fix.shape[0])
    affined_mov = cv2.warpAffine(mov, affine, output_size, flags=cv2.INTER_CUBIC)

    return fix_spots_t, mov_spots_t, affine, affined_mov


def draw_points(num, fix_png_path, fix_spots, mov_png_path, mov_spots, point_size=5):
    """
    Feature points are visualised onto the original image for ease of viewing

    parameter
    ---------
    num: int
        Aligned rounds for easy naming when saving images

    """

    fix_png = Image.open(fix_png_path)
    draw_fix = ImageDraw.Draw(fix_png)
    for fix_coord in fix_spots:
        x, y = fix_coord
        bbox = (x - point_size, y - point_size, x + point_size, y + point_size)
        draw_fix.ellipse(bbox, fill='red')
    # fix_png.save('./data2/registration/round1_'+str(num)+'_with_feature_points.png')

    mov_png = Image.open(mov_png_path)
    draw_mov = ImageDraw.Draw(mov_png)
    for mov_coord in mov_spots:
        x, y = mov_coord
        bbox = (x - point_size, y - point_size, x + point_size, y + point_size)
        draw_mov.ellipse(bbox, fill='red')
    # mov_png.save('./data2/registration/round' + str(num) + 'with_feature_points.png')

    width, height = fix_png.size
    new_image = Image.new('RGB', (width, 2*height))
    new_image.paste(fix_png, (0, 0))
    new_image.paste(mov_png, (0, height))
    new_image.save('./data2/registration/round' + str(num) + 'with_feature_points.png')

    return


def draw_dapi_points(png_path, spots, save_path, point_size=3):
    png = Image.open(png_path)
    draw = ImageDraw.Draw(png)
    for coord in spots:
        x, y = coord
        bbox = (x - point_size, y - point_size, x + point_size, y + point_size)
        draw.ellipse(bbox, fill='red')
    png.save(save_path)
    return


def apply_transform(img, affine):
    """
    For affine transformation of fluorescent signal points

    parameter
    ---------
    img: 2d-array
    affine: 2d-array 2x3
        affine transformation matrix

    """

    print('transforming', flush=True)
    output_size = (img.shape[1], img.shape[0])
    affined_img = cv2.warpAffine(img, affine, output_size, flags=cv2.INTER_LANCZOS4)
    return affined_img
