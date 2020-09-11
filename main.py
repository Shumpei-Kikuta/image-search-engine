import pdb
import cv2

import glob

from sklearn.metrics.pairwise import cosine_similarity

MAX = 100000000
IMG_SIZE = (200, 200)


def main():
    akaze = cv2.AKAZE_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    dicts = {}
    img_path_lists = glob.glob('img/*.jpg')
    for img_path in img_path_lists:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, IMG_SIZE)
        # 特徴量の検出と特徴量ベクトルの計算
        kp1, des1 = akaze.detectAndCompute(img, None)
        dicts[img_path] = des1

    print('start!')
    test_img = 'IMG_1130.JPG'
    img = cv2.imread(test_img, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, IMG_SIZE)
    # 特徴量の検出と特徴量ベクトルの計算
    kp1, target_des = akaze.detectAndCompute(img, None)

    min_distance = MAX
    min_path = None
    for img_path, features in dicts.items():
        matches = bf.match(target_des, features)
        dist = [m.distance for m in matches]
        ret = sum(dist) / len(dist)
        print(img_path, ret)
        if ret < min_distance:
            min_distance = ret
            min_path = img_path
    print(min_path)


if __name__ == '__main__':
    main()
