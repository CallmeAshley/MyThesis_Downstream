import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

# 파일을 불러올 폴더와 저장할 폴더 지정
source_dir = '/mai_nas/LSH/MRSpineSeg_Challenge_SMU/dataset/2D/train_ideal/MR'
target_dir = '/mai_nas/LSH/MRSpineSeg_Challenge_SMU/dataset/2D/train_ideal_png/MR'



# 저장할 폴더가 없으면 생성
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# source_dir에서 .npy 파일 목록을 가져옴
for file_name in os.listdir(source_dir):
    if file_name.endswith('.npy'):
        # 파일 경로 생성
        file_path = os.path.join(source_dir, file_name)
        
        # .npy 파일 불러오기
        data = np.flipud(np.load(file_path))


        # 각 채널에 동일한 데이터를 사용하여 RGB 이미지 생성
        # data_rgb = np.stack([data]*3, axis=-1)
        
        # 이미지로 저장
        plt.imshow(data, cmap='gray')
        plt.axis('off')  # 축 숨기기
        
        # 파일 이름에서 .npy를 제거하고 .png로 변경
        png_file_name = file_name[:-4] + '.png'
        png_file_path = os.path.join(target_dir, png_file_name)
        
        plt.savefig(png_file_path, bbox_inches='tight', pad_inches=0)
        # plt.savefig(png_file_path)
        plt.close()  # 현재 플롯 닫기
        
        

# img = cv2.imread('/mai_nas/LSH/MRSpineSeg_Challenge_SMU/dataset/2D/test1_ideal_png/MR/Case65_8.png')
# print(' ')
