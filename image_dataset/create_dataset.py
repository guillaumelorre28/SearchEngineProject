import os
import cv2 as cv
from collections import defaultdict
import json


def list_gt_files(gt_path):
    list_files = []
    list_monuments = []
    for file in os.listdir(gt_path):
        if file.endswith('good.txt'):
            if file[:-11] not in list_monuments:
                list_files.append(file)
                list_monuments.append(file[:-11])
    return list_files


def compute_descriptors(image_path):

    sift = cv.xfeatures2d.SIFT_create()
    img = cv.imread(image_path)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    kp, des = sift.detectAndCompute(gray, None)
    pts = [k.pt for k in kp]
    return pts, des


def read_gt_file(gt_filename):

    images_list = []
    gt_file = open(gt_filename, "r")
    for line in gt_file.readlines():
        images_list.append(line.strip())
    return images_list


def create_dataset(gt_path, image_path):

    dataset = {}
    gt_files_list = list_gt_files(gt_path)
    for gt_file in gt_files_list:
        dataset_image = defaultdict(lambda: [])
        images_list = read_gt_file(os.path.join(gt_path, gt_file))
        for image in images_list:
            try:
                pts, des = compute_descriptors(os.path.join(image_path, image + '.jpg'))
                des = des.tolist()
                dataset_image[image].append((pts, des))
                dataset[gt_file[:-11]].append((pts, des))
            except:
                print('ERROR: ' + gt_file + '  ' + image)

        dataset[gt_file[:-11]] = dataset_image
    return dataset


if __name__ == '__main__':

    folder_path = '/home/guillaume/Desktop/image_dataset/image_dataset'
    gt_path = os.path.join(folder_path, 'gt_files_170407')
    images_path = os.path.join(folder_path, 'oxbuild_images')

    dataset = create_dataset(gt_path, images_path)

    for key in dataset:
        with open('dataset_images' + key + '.json', 'w') as outfile:
            json.dump(dataset[key], outfile)
