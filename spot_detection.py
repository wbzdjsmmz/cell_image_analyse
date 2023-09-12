from PIL import Image, ImageDraw
from skimage.feature import blob_log


def get_feature_spots(image, min_sigma=5, max_sigma=12, threshold=0.1, exclude_border=50):
    num_sigma = max_sigma - min_sigma
    spots = blob_log(image, min_sigma, max_sigma, num_sigma, threshold=threshold, exclude_border=exclude_border)
    return spots


def draw_dapi_points(png_path, spots, save_path, color):
    point_size = 1
    png = Image.open(png_path)
    draw = ImageDraw.Draw(png)
    for coord in spots:
        x, y = coord
        bbox = (x - point_size, y - point_size, x + point_size, y + point_size)
        draw.ellipse(bbox, fill=color)
        # draw.point((x, y), fill=color)
    png.save(save_path)
    return
