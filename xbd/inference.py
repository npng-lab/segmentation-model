# 해당 코드는 @jihoon2819 작성한 ipynb 파일을 python 코드로 변환했습니다.
import numpy as np
import tensorflow as tf
import os
import math
import PIL
import cv2
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, array_to_img
from PIL import ImageOps , Image as PILImage
from IPython.display import Image, display

# .h5 파일로부터 모델 로드
model = load_model(userdata.get('AI_URL'))

input_dir = 'test/images'
img_size = (1024,1024)
batch_size = 1

input_img_paths = sorted([ os.path.join(input_dir, fname)
                           for fname in os.listdir(input_dir)
                           if fname.endswith('post_disaster.png')])

test_input_img_paths = [input_img_paths[12]]

# 추론용 데이터 제너레이터 클래스 정의
class xBDInference(keras.utils.Sequence):
    def __init__(self, batch_size, img_size, input_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths

    def __len__(self):
        return math.ceil(len(self.input_img_paths) / self.batch_size)

    def __getitem__(self, idx):
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i: i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype='float32')
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size)
            x[j] = img
        return x

test_gen = xBDInference(batch_size, img_size, test_input_img_paths)

test_preds = model.predict(test_gen)

def dfs(matrix, visited, x, y):
    # 이동할 수 있는 방향 (상, 하, 좌, 우)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    stack = [[x, y]]
    component = [[x, y]]

    while stack:
        cx, cy = stack.pop()
        for dx, dy in directions:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < len(matrix) and 0 <= ny < len(matrix[0]) and not visited[nx][ny] and matrix[nx][ny] > 0.5:
                visited[nx][ny] = True
                stack.append([nx, ny])
                component.append([nx, ny])

    return component

def find_components(matrix):
    visited = np.zeros_like(matrix, dtype=bool)
    components = []

    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] > 0.5 and not visited[i][j]:
                visited[i][j] = True
                component = dfs(matrix, visited, i, j)
                components.append(component)

    return components

def find_boundaries(component):
    top = max(component, key=lambda x: x[1])
    top = top[1]
    bottom = min(component, key=lambda x: x[1])
    bottom= bottom[1]
    left = min(component, key=lambda x: x[0])
    left = left[0]
    right = max(component, key=lambda x: x[0])
    right = right[0]
    width= right - left+1
    height=  top-bottom+1
    instance_size = [width, height]
    vertical_center =  float((top + bottom) / 2 )
    horizontal_center =float((right +left) / 2 )
    position_xy=[horizontal_center,vertical_center]
    return instance_size,position_xy,left,top,right,bottom


# 연결된 객체들 찾기
components = find_components(np.array(test_preds[0]))

# 1차원 배열로 변환
flattened_components = [component for component in components]
# 객체의 정보
matrix_components = [find_boundaries(component) for component in components]

#mask_url 앞에 {}로 경로 수정해서 넣어주시면 됩니다
instance_info=[]
for i in range(len(matrix_components)):
    example={"id": f'building{i}',
             "mask_url": f'building{i}.png',
             "box_coordinates": [(matrix_components[i][2], matrix_components[i][3]), (matrix_components[i][4], matrix_components[i][3]), (matrix_components[i][4], matrix_components[i][5]), (matrix_components[i][2], matrix_components[i][5])]
             }
    instance_info.append(example)

# instance info에 객체 정보가 저장 됩니다
print(instance_info)

save_dir = userdata.get('DB_URL') # 경로 변경 필요
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

index=6
def save_sperate_image(matrix_component,flattened_component):
  width=matrix_component[0][0]
  height=matrix_component[0][1]
  im = PILImage.open(test_input_img_paths[0])
  # print(flattened_component)
  # print(f'left{matrix_component[2]},upper {matrix_component[5]},right{matrix_component[4]}, lower{matrix_component[3]}')
  cropbox=(matrix_component[5],matrix_component[2],matrix_component[3]+1,matrix_component[4]+1)
  cropped=im.crop(cropbox)
  np_cropped = np.array(cropped)
  # display(cropped)
  red_np_cropped = np_cropped[:,:,0]
  green_np_cropped = np_cropped[:,:,1]
  blue_np_cropped = np_cropped[:,:,2]
  image_array = np.zeros((width, height), dtype=np.uint8)
  # print(image_array)
  for coord in flattened_component:
      x, y = coord
      # print(x,y)
      if 0 <= (x - matrix_component[2]) and 0 <= (y - matrix_component[5]) :
          image_array[x - matrix_component[2] ] [y - matrix_component[5]]= 155  # 값을 1로 설정, 필요에 따라 다른 값으로 설정 가능
          # print(image_array)
  for i in range(len(image_array)):
      for j in range(len(image_array[0])):
          if image_array[i][j] == 0:
              red_np_cropped[i][j] =0
              green_np_cropped[i][j] =0
              blue_np_cropped[i][j] =0
  image = PILImage.fromarray(image_array)

  # 이미지를 'L' 모드로 변환하여 8비트 흑백 이미지로 저장
  image = image.convert('L')
  red_image = PILImage.fromarray(red_np_cropped.astype(np.uint8), mode='L')
  green_image = PILImage.fromarray(green_np_cropped.astype(np.uint8), mode='L')
  blue_image = PILImage.fromarray(blue_np_cropped.astype(np.uint8), mode='L')
  mergeImage=(PILImage.merge("RGB", (red_image, green_image, blue_image)))
  # print(red_np_cropped)
  # print(green_np_cropped)
  # print(blue_np_cropped)
  display(mergeImage)
  mergeImage.save("/content/drive/MyDrive/xBD/instance_img/${matrix_component[30]}.png",format="png")
  # //이터러블 하게 바꾸기 이름문저장 할때 문제 해결하기

save_sperate_image(matrix_components[7],flattened_components[7])
for i in range(len(matrix_components)):
    save_sperate_image(matrix_components[i],flattened_components[i])
