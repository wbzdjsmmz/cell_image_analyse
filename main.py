import tifffile
import numpy as np
from registration import (Registration, draw_points,
                          apply_transform)
from stardist.models import StarDist2D
from csbdeep.utils import normalize
import matplotlib.pyplot as plt
import cv2
from spot_detection import get_feature_spots, draw_dapi_points
import pandas as pd
from draw import draw_cell


# # ========================================== Part1 registration ================================================ #
# # Import DAPI staining data
# round1 = tifffile.imread('./data2/registration/1_Dapi.tif')
# round2 = tifffile.imread('./data2/registration/2_Dapi.tif')
# round3 = tifffile.imread('./data2/registration/3_Dapi.tif')
# round4 = tifffile.imread('./data2/registration/4_Dapi.tif')
#
# print('Registering round1 DAPI-stained image with round2 DAPI-stained image')
# round1_spots, round2_spots, affine2_1, affined_round2 = Registration(round1, round2, match_threshold=150000000)
# tifffile.imwrite('./data2/registration/affined_round2.tiff', affined_round2)
# np.savetxt('./data2/registration/affine2_1.mat', affine2_1)
# draw_points(2, './data2/registration/1_Dapi.png', round1_spots, './data2/registration/2_Dapi.png', round2_spots)
#
# print('Registering round1 DAPI-stained image with round3 DAPI-stained image')
# round1_spots_, round3_spots, affine3_1, affined_round3 = Registration(round1, round3)
# tifffile.imwrite('./data2/registration/affined_round3.tiff', affined_round3)
# np.savetxt('./data2/registration/affine3_1.mat', affine3_1)
# draw_points(3, './data2/registration/1_Dapi.png', round1_spots_, './data2/registration/3_Dapi.png', round3_spots)
#
# print('Registering round1 DAPI-stained image with round4 DAPI-stained image')
# round1_spots__, round4_spots, affine4_1, affined_round4 = Registration(round1, round4, match_threshold=150000000)
# tifffile.imwrite('./data2/registration/affined_round4.tiff', affined_round4)
# np.savetxt('./data2/registration/affine4_1.mat', affine4_1)
# draw_points(4, './data2/registration/1_Dapi.png', round1_spots__, './data2/registration/4_Dapi.png', round4_spots)

# # ========================================= Part2 apply registration ============================================ #
# print('Aligning the three fluorescences from the second round of imaging to the first round ')
# affine2_1 = np.loadtxt('./data2/registration/affine2_1.mat')
# # Aldh1a1
# Aldh1a1 = tifffile.imread('./data2/spots/Aldh1a1.tif')
# affined_Aldh1a1 = apply_transform(Aldh1a1, affine2_1)
# tifffile.imwrite('./data2/registered_spots/affined_Aldh1a1.tif', affined_Aldh1a1)
# # Sncg
# Sncg = tifffile.imread('./data2/spots/Sncg.tif')
# affined_Sncg = apply_transform(Sncg, affine2_1)
# tifffile.imwrite('./data2/registered_spots/affined_Sncg.tif', affined_Sncg)
# # Calb2
# Calb2 = tifffile.imread('./data2/spots/Calb2.tif')
# affined_Calb2 = apply_transform(Calb2, affine2_1)
# tifffile.imwrite('./data2/registered_spots/affined_Calb2.tif', affined_Calb2)
#
# print('Aligning the three fluorescences from the third round of imaging to the first round ')
# affine3_1 = np.loadtxt('./data2/registration/affine3_1.mat')
# # Ndnf
# Ndnf = tifffile.imread('./data2/spots/Ndnf.tif')
# affined_Ndnf = apply_transform(Ndnf, affine3_1)
# tifffile.imwrite('./data2/registered_spots/affined_Ndnf.tif', affined_Ndnf)
# # Sox6
# Sox6 = tifffile.imread('./data2/spots/Sox6.tif')
# affined_Sox6 = apply_transform(Sox6, affine3_1)
# tifffile.imwrite('./data2/registered_spots/affined_Sox6.tif', affined_Sox6)
# # Calb1
# Calb1 = tifffile.imread('./data2/spots/Calb1.tif')
# affined_Calb1 = apply_transform(Calb1, affine3_1)
# tifffile.imwrite('./data2/registered_spots/affined_Calb1.tif', affined_Calb1)
#
# print('Aligning the three fluorescences from the fourth round of imaging to the first round ')
# affine4_1 = np.loadtxt('./data2/registration/affine4_1.mat')
# # Slc6a3
# Slc6a3 = tifffile.imread('./data2/spots/Slc6a3.tif')
# affined_Slc6a3 = apply_transform(Slc6a3, affine4_1)
# tifffile.imwrite('./data2/registered_spots/affined_Slc6a3.tif', affined_Slc6a3)
# # Th
# Th = tifffile.imread('./data2/spots/Th.tif')
# affined_Th = apply_transform(Th, affine4_1)
# tifffile.imwrite('./data2/registered_spots/affined_Th.tif', affined_Th)
# # Vglut2
# Vglut2 = tifffile.imread('./data2/spots/Vglut2.tif')
# affined_Vglut2 = apply_transform(Vglut2, affine4_1)
# tifffile.imwrite('./data2/registered_spots/affined_Vglut2.tif', affined_Vglut2)

# ============================================ Part3 cell segamention =============================================== #
# SNC = tifffile.imread('./data2/segmention/SNC.tif')
# model = StarDist2D.from_pretrained('2D_paper_dsb2018')
# original_labels, _ = model.predict_instances(normalize(SNC), prob_thresh=0.54, nms_thresh=0.6)
# tifffile.imwrite('./data2/segmention/original_labels.tif', original_labels)
#
# labels = tifffile.imread('./data2/segmention/original_labels.tif')
#
# # Calculate histogram
# flattened_labels = labels.ravel()
# histogram, bins, _ = plt.hist(flattened_labels, 400, [1, 400])
#
# # Customised thresholds, which can be adjusted according to the actual situation
# high_threshold_freq = 2000
# high_freq_indices = np.where(histogram > high_threshold_freq)[0]
# low_threshold_freq = 200
# low_freq_indices = np.where(histogram < low_threshold_freq)[0]
# for index in high_freq_indices:
#     labels[labels == (index + 1)] = 0
# for index in low_freq_indices:
#     labels[labels == (index + 1)] = 0
# # Observe the image with the naked eye and manually pick cells for deletion
# labels[labels == 330] = 0
# labels[labels == 331] = 0
# labels[labels == 258] = 0
# labels[labels == 277] = 0
# labels[labels == 305] = 210
# labels[labels == 316] = 0
# labels[labels == 310] = 0
# labels[labels == 131] = 236
# labels[labels == 288] = 236
# tifffile.imwrite('./data2/segmention/final_labels.tif', labels)
# flattened_labels = labels.ravel()
# histogram, bins, _ = plt.hist(flattened_labels, 400, [1, 400])
#
# # Plot the histogram
# plt.title('Cell to delete')
# plt.xlabel('Pixel Value')
# plt.ylabel('Frequency')
# plt.ylim(0, 3000)
# plt.show()
# #
# # Reordering cell profile labels
# seg_result = tifffile.imread('./data2/segmention/final_labels.tif')
# # 初始化标记图像，将所有像素初始化为0
# labels = np.zeros_like(seg_result)
#
# # 定义当前标记
# current_label = 1
#
# # 遍历灰度值图像的每个像素
# for y in range(seg_result.shape[0]):
#     for x in range(seg_result.shape[1]):
#         # 如果当前像素还没有被标记
#         if labels[y, x] == 0 and seg_result[y, x] != 0:
#             # 使用当前标记对所有相同灰度值的像素进行标记
#             mask = (seg_result == seg_result[y, x])
#             labels[mask] = current_label
#             # 增加下一个标记
#             current_label += 1
# cell_num = np.max(labels)
# tifffile.imwrite('./data2/segmention/seg_result.tif', labels)

# # ============================================== Spot detection ============================================= #
# seg_result = tifffile.imread('./data2/segmention/seg_result.tif')
# cell_num = np.max(seg_result)
# point_result = np.zeros((cell_num, 9))
#
# # The following spot images are on fiji with some preprocessing (e.g., open ops, top hat ops, etc.)
# Aldh1a1_path = './data2/spot_detection_result/Aldh1a1/tophat_affined_Aldh1a1.tif'
# Aldh1a1_png_path = './data2/spot_detection_result/Aldh1a1/tophat_affined_Aldh1a1.png'
# Aldh1a1_with_spots_path = './data2/spot_detection_result/Aldh1a1/tophat_affined_Aldh1a1_spots.png'
#
# Calb1_path = './data2/spot_detection_result/Calb1/tophat_affined_Calb1.tif'
# Calb1_png_path = './data2/spot_detection_result/Calb1/tophat_affined_Calb1.png'
# Calb1_with_spots_path = './data2/spot_detection_result/Calb1/tophat_affined_Calb1_spots.png'
#
# Calb2_path = './data2/spot_detection_result/Calb2/tophat_affined_Calb2.tif'
# Calb2_png_path = './data2/spot_detection_result/Calb2/tophat_affined_Calb2.png'
# Calb2_with_spots_path = './data2/spot_detection_result/Calb2/tophat_affined_Calb2_spots.png'
#
# Ndnf_path = './data2/spot_detection_result/Ndnf/manu_affined_Ndnf.tif'
# Ndnf_png_path = './data2/spot_detection_result/Ndnf/manu_affined_Ndnf.png'
# Ndnf_with_spots_path = './data2/spot_detection_result/Ndnf/manu_affined_Ndnf_spots.png'
#
# Slc6a3_path = './data2/spot_detection_result/Slc6a3/tophat_affined_Slc6a3.tif'
# Slc6a3_png_path = './data2/spot_detection_result/Slc6a3/tophat_affined_Slc6a3.png'
# Slc6a3_with_spots_path = './data2/spot_detection_result/Slc6a3/tophat_affined_Slc6a3_spots.png'
#
# Sncg_path = './data2/spot_detection_result/Sncg/manu_affined_Sncg.tif'
# Sncg_png_path = './data2/spot_detection_result/Sncg/manu_affined_Sncg.png'
# Sncg_with_spots_path = './data2/spot_detection_result/Sncg/manu_affined_Sncg_spots.png'
#
# Sox6_path = './data2/spot_detection_result/Sox6/manu_affined_Sox6.tif'
# Sox6_png_path = './data2/spot_detection_result/Sox6/manu_affined_Sox6.png'
# Sox6_with_spots_path = './data2/spot_detection_result/Sox6/manu_affined_Sox6_spots.png'
#
# Th_path = './data2/spot_detection_result/Th/tophat_affined_Th.tif'
# Th_png_path = './data2/spot_detection_result/Th/tophat_affined_Th.png'
# Th_with_spots_path = './data2/spot_detection_result/Th/tophat_affined_Th_spots.png'
#
# Vglut2_path = './data2/spot_detection_result/Vglut2/manu_affined_Vglut2.tif'
# Vglut2_png_path = './data2/spot_detection_result/Vglut2/manu_affined_Vglut2.png'
# Vglut2_with_spots_path = './data2/spot_detection_result/Vglut2/manu_affined_Vglut2_spots.png'
#
# # ----------------------------------------------Aldh1a1---------------------------------------------- #
# Aldh1a1 = tifffile.imread(Aldh1a1_path)
# Aldh1a1_spots = get_feature_spots(Aldh1a1, min_sigma=1, max_sigma=3, threshold=0.05, exclude_border=0)
#
# Aldh1a1_spots = Aldh1a1_spots[:, :2]
# Aldh1a1_spots[:, [0, 1]] = Aldh1a1_spots[:, [1, 0]]
# draw_dapi_points(Aldh1a1_png_path, Aldh1a1_spots, save_path=Aldh1a1_with_spots_path, color='red')
#
# new_Aldh1a1_spots = []
# # Next, plot the points onto the graph and count the number of fluorescent points for each type of fluorescence
# for i in range(Aldh1a1_spots.shape[0]):
#     value = seg_result[int(Aldh1a1_spots[i][1]), int(Aldh1a1_spots[i][0])]
#     if value != 0:
#         point_result[value-1, 0] += 1
#         new_Aldh1a1_spots.append(Aldh1a1_spots[i])
#
# new_Aldh1a1_spots = np.array(new_Aldh1a1_spots)
# draw_dapi_points('./data2/spot_detection_result/result/seg_result.jpg', new_Aldh1a1_spots, './data2/spot_detection_result/result/seg_result1.jpg', color=(0, 255, 0))
#
# # ----------------------------------------------Calb1---------------------------------------------- #
# Calb1 = tifffile.imread(Calb1_path)
# Calb1_spots = get_feature_spots(Calb1, min_sigma=1, max_sigma=3, threshold=0.05, exclude_border=0)
#
# Calb1_spots = Calb1_spots[:, :2]
# Calb1_spots[:, [0, 1]] = Calb1_spots[:, [1, 0]]
# draw_dapi_points(Calb1_png_path, Calb1_spots, save_path=Calb1_with_spots_path, color='red')
#
# new_Calb1_spots = []
# # Next, plot the points onto the graph and count the number of fluorescent points for each type of fluorescence
# for i in range(Calb1_spots.shape[0]):
#     value = seg_result[int(Calb1_spots[i][1]), int(Calb1_spots[i][0])]
#     if value != 0:
#         point_result[value-1, 1] += 1
#         new_Calb1_spots.append(Calb1_spots[i])
#
# new_Calb1_spots = np.array(new_Calb1_spots)
# draw_dapi_points('./data2/spot_detection_result/result/seg_result1.jpg', new_Calb1_spots, './data2/spot_detection_result/result/seg_result2.jpg', color=(255, 255, 0))
#
# # ----------------------------------------------Calb2---------------------------------------------- #
# Calb2 = tifffile.imread(Calb2_path)
# Calb2_spots = get_feature_spots(Calb2, min_sigma=1, max_sigma=3, threshold=0.02, exclude_border=0)
#
# Calb2_spots = Calb2_spots[:, :2]
# Calb2_spots[:, [0, 1]] = Calb2_spots[:, [1, 0]]
# draw_dapi_points(Calb2_png_path, Calb2_spots, save_path=Calb2_with_spots_path, color='red')
#
# new_Calb2_spots = []
# # Next, plot the points onto the graph and count the number of fluorescent points for each type of fluorescence
# for i in range(Calb2_spots.shape[0]):
#     value = seg_result[int(Calb2_spots[i][1]), int(Calb2_spots[i][0])]
#     if value != 0:
#         point_result[value-1, 2] += 1
#         new_Calb2_spots.append(Calb2_spots[i])
#
# new_Calb2_spots = np.array(new_Calb2_spots)
# draw_dapi_points('./data2/spot_detection_result/result/seg_result2.jpg', new_Calb2_spots, './data2/spot_detection_result/result/seg_result3.jpg', color=(0, 255, 255))
#
# # ------------------------------------------------Ndnf ----------------------------------------------- #
# Ndnf = tifffile.imread(Ndnf_path)
# Ndnf_spots = get_feature_spots(Ndnf, min_sigma=1, max_sigma=10, threshold=0.3, exclude_border=0)
#
# Ndnf_spots = Ndnf_spots[:, :2]
# Ndnf_spots[:, [0, 1]] = Ndnf_spots[:, [1, 0]]
# draw_dapi_points(Ndnf_png_path, Ndnf_spots, Ndnf_with_spots_path, color='red')
#
# new_Ndnf_spots = []
# # Next, plot the points onto the graph and count the number of fluorescent points for each type of fluorescence
# for i in range(Ndnf_spots.shape[0]):
#     value = seg_result[int(Ndnf_spots[i][1]), int(Ndnf_spots[i][0])]
#     if value != 0:
#         point_result[value-1, 3] += 1
#         new_Ndnf_spots.append(Ndnf_spots[i])
#
# new_Ndnf_spots = np.array(new_Ndnf_spots)
# draw_dapi_points('./data2/spot_detection_result/result/seg_result3.jpg', new_Ndnf_spots, './data2/spot_detection_result/result/seg_result4.jpg', color=(255, 0, 0))
#
# # ------------------------------------------------Slc6a3 ----------------------------------------------- #
# Slc6a3 = tifffile.imread(Slc6a3_path)
# Slc6a3_spots = get_feature_spots(Slc6a3, min_sigma=1, max_sigma=10, threshold=0.3, exclude_border=0)
#
# Slc6a3_spots = Slc6a3_spots[:, :2]
# Slc6a3_spots[:, [0, 1]] = Slc6a3_spots[:, [1, 0]]
# draw_dapi_points(Slc6a3_png_path, Slc6a3_spots, Slc6a3_with_spots_path, color='red')
#
# new_Slc6a3_spots = []
# # Next, plot the points onto the graph and count the number of fluorescent points for each type of fluorescence
# for i in range(Slc6a3_spots.shape[0]):
#     value = seg_result[int(Slc6a3_spots[i][1]), int(Slc6a3_spots[i][0])]
#     if value != 0:
#         point_result[value-1, 4] += 1
#         new_Slc6a3_spots.append(Slc6a3_spots[i])
#
# new_Slc6a3_spots = np.array(new_Slc6a3_spots)
# draw_dapi_points('./data2/spot_detection_result/result/seg_result4.jpg', new_Slc6a3_spots, './data2/spot_detection_result/result/seg_result5.jpg', color=(255, 0, 255))
#
# # ------------------------------------------------Sncg ----------------------------------------------- #
# Sncg = tifffile.imread(Sncg_path)
# Sncg_spots = get_feature_spots(Sncg, min_sigma=1, max_sigma=10, threshold=0.2, exclude_border=0)
#
# Sncg_spots = Sncg_spots[:, :2]
# Sncg_spots[:, [0, 1]] = Sncg_spots[:, [1, 0]]
# draw_dapi_points(Sncg_png_path, Sncg_spots, Sncg_with_spots_path, color='red')
#
# new_Sncg_spots = []
# # Next, plot the points onto the graph and count the number of fluorescent points for each type of fluorescence
# for i in range(Sncg_spots.shape[0]):
#     value = seg_result[int(Sncg_spots[i][1]), int(Sncg_spots[i][0])]
#     if value != 0:
#         point_result[value-1, 5] += 1
#         new_Sncg_spots.append(Sncg_spots[i])
#
# new_Sncg_spots = np.array(new_Sncg_spots)
# draw_dapi_points('./data2/spot_detection_result/result/seg_result5.jpg', new_Sncg_spots, './data2/spot_detection_result/result/seg_result6.jpg', color=(0, 0, 255))
#
# # ------------------------------------------------Sox6 ----------------------------------------------- #
# Sox6 = tifffile.imread(Sox6_path)
# Sox6_spots = get_feature_spots(Sox6, min_sigma=1, max_sigma=10, threshold=0.3, exclude_border=0)
#
# Sox6_spots = Sox6_spots[:, :2]
# Sox6_spots[:, [0, 1]] = Sox6_spots[:, [1, 0]]
# draw_dapi_points(Sox6_png_path, Sox6_spots, Sox6_with_spots_path, color='red')
#
# new_Sox6_spots = []
# # Next, plot the points onto the graph and count the number of fluorescent points for each type of fluorescence
# for i in range(Sox6_spots.shape[0]):
#     value = seg_result[int(Sox6_spots[i][1]), int(Sox6_spots[i][0])]
#     if value != 0:
#         point_result[value-1, 6] += 1
#         new_Sox6_spots.append(Sox6_spots[i])
#
# new_Sox6_spots = np.array(new_Sox6_spots)
# draw_dapi_points('./data2/spot_detection_result/result/seg_result6.jpg', new_Sox6_spots, './data2/spot_detection_result/result/seg_result7.jpg', color=(255, 192, 203))
#
# # ------------------------------------------------Th ----------------------------------------------- #
# Th = tifffile.imread(Th_path)
# Th_spots = get_feature_spots(Th, min_sigma=1, max_sigma=4, threshold=0.03, exclude_border=0)
#
# Th_spots = Th_spots[:, :2]
# Th_spots[:, [0, 1]] = Th_spots[:, [1, 0]]
# draw_dapi_points(Th_png_path, Th_spots, Th_with_spots_path, color='red')
#
# new_Th_spots = []
# # Next, plot the points onto the graph and count the number of fluorescent points for each type of fluorescence
# for i in range(Th_spots.shape[0]):
#     value = seg_result[int(Th_spots[i][1]), int(Th_spots[i][0])]
#     if value != 0:
#         point_result[value-1, 7] += 1
#         new_Th_spots.append(Th_spots[i])
#
# new_Th_spots = np.array(new_Th_spots)
# draw_dapi_points('./data2/spot_detection_result/result/seg_result7.jpg', new_Th_spots, './data2/spot_detection_result/result/seg_result8.jpg', color=(165, 42, 42))
#
# # ------------------------------------------------Vglut2 ----------------------------------------------- #
# Vglut2 = tifffile.imread(Vglut2_path)
# Vglut2_spots = get_feature_spots(Vglut2, min_sigma=1, max_sigma=3, threshold=0.3, exclude_border=0)
# Vglut2_spots = Vglut2_spots[:, :2]
# Vglut2_spots[:, [0, 1]] = Vglut2_spots[:, [1, 0]]
# draw_dapi_points(Vglut2_png_path, Vglut2_spots, Vglut2_with_spots_path, color='red')
#
# new_Vglut2_spots = []
# # Next, plot the points onto the graph and count the number of fluorescent points for each type of fluorescence
# for i in range(Vglut2_spots.shape[0]):
#     value = seg_result[int(Vglut2_spots[i][1]), int(Vglut2_spots[i][0])]
#     if value != 0:
#         point_result[value-1, 8] += 1
#         new_Vglut2_spots.append(Vglut2_spots[i])
#
# new_Vglut2_spots = np.array(new_Vglut2_spots)
# draw_dapi_points('./data2/spot_detection_result/result/seg_result8.jpg', new_Vglut2_spots, './data2/spot_detection_result/result/seg_result9.jpg', color=(255, 165, 0))
#
#
# column_labels = ["Aldh1a1", "Calb1", "Calb2", "Ndnf", "Slc6a3", "Sncg", "Sox6", "Th", "Vglut2"]
# row_labels = [f'Cell {i}' for i in range(1, cell_num + 1)]
# df = pd.DataFrame(point_result, index=row_labels, columns=column_labels)
# df.to_excel('./data2/spot_detection_result/result/point_result.xlsx', index=True)

# ========================================== Plotting clustering results =========================================== #
seg_result = tifffile.imread('./data2/segmention/seg_result.tif')

cluster = pd.read_csv('./cluster1.csv')
categories = cluster['seurat_clusters'].to_numpy()
index = cluster.iloc[:, 0]
index = [int(item[4:]) for item in index]

draw_cell(index, categories, seg_result)
