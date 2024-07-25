# 해당 코드는 @jihoon2819 작성한 ipynb 파일을 python 코드로 변환했습니다.
import os
import math
import queue
import PIL
import random
import numpy as np
from dataclasses import dataclass, asdict
from PIL import ImageOps , Image as PILImage
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from typing import List, Tuple, Dict, Any
from uuid import UUID, uuid4

save_dir = userdata.get('DB_URL')
if not os.path.exists(save_dir):
  os.makedirs(save_dir)

# .h5 파일로부터 모델 로드
model = load_model(userdata.get('AI_URL'))

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

def save_sperate_image(instance_id: UUID, input_image_path: str, matrix_component,flattened_component):
  width=matrix_component[0][0]
  height=matrix_component[0][1]
  im = PILImage.open(input_image_path)

  cropbox=(matrix_component[5],matrix_component[2],matrix_component[3]+1,matrix_component[4]+1)
  cropped=im.crop(cropbox)
  cropped=cropped.convert('RGBA')
  np_cropped = np.array(cropped)

  red_np_cropped = np_cropped[:,:,0]
  green_np_cropped = np_cropped[:,:,1]
  blue_np_cropped = np_cropped[:,:,2]
  alpha_np_cropped = np_cropped[:,:,3]
  image_array = np.zeros((width, height), dtype=np.uint8)

  for coord in flattened_component:
    x, y = coord

    if 0 <= (x - matrix_component[2]) and 0 <= (y - matrix_component[5]) :
      image_array[x - matrix_component[2] ] [y - matrix_component[5]]= 155  # 값을 1로 설정, 필요에 따라 다른 값으로 설정 가능

  for i in range(len(image_array)):
    for j in range(len(image_array[0])):
      if image_array[i][j] == 0:
        red_np_cropped[i][j] =0
        green_np_cropped[i][j] =0
        blue_np_cropped[i][j] =0
        alpha_np_cropped[i][j]=0
  image = PILImage.fromarray(image_array)

  image = image.convert('L')
  red_image = PILImage.fromarray(red_np_cropped.astype(np.uint8), mode='L')
  green_image = PILImage.fromarray(green_np_cropped.astype(np.uint8), mode='L')
  blue_image = PILImage.fromarray(blue_np_cropped.astype(np.uint8), mode='L')
  alpha_image = PILImage.fromarray(alpha_np_cropped.astype(np.uint8), mode='L')
  mergeImage=(PILImage.merge("RGBA", (red_image, green_image, blue_image,alpha_image)))
  mergeImage.save(f"{save_dir}/{instance_id}.png",format="png")

def save_sperate_background_image(instance_id: UUID, input_image_path: str,flattened_component):

  im = PILImage.open(input_image_path)
  width,height=im.size
  box_padding=0
  cropbox=(0,0,width,height)
  cropped=im.crop(cropbox)
  np_cropped = np.array(cropped)
  # print('check point 1')
  red_np_cropped = np_cropped[:,:,0]
  green_np_cropped = np_cropped[:,:,1]
  blue_np_cropped = np_cropped[:,:,2]
  image_array = np.zeros((width, height), dtype=np.float64)
  red_np_cropped=red_np_cropped.astype(np.float64)
  green_np_cropped=green_np_cropped.astype(np.uint64)
  blue_np_cropped=blue_np_cropped.astype(np.uint64)

  forward_direction=range(box_padding,len(image_array)-box_padding)
  backward_direction=range(len(image_array)-box_padding-1,box_padding+1,-1)
  downward_direction=range(box_padding,len(image_array[0])-box_padding)
  upward_direction=range(len(image_array[0])-box_padding-1,box_padding+1,-1)

  for component in flattened_component:
    for coord in component:
      x, y = coord
      if 0 <= (x) and 0 <= (y) :
        image_array[x] [y]= 155  # 값을 1로 설정, 필요에 따라 다른 값으로 설정 가능
  for i in forward_direction:
    for j in downward_direction:
      if image_array[i][j] != 0:
        red_np_cropped[i][j] =0
        green_np_cropped[i][j] =0
        blue_np_cropped[i][j] =0

 ## 패딩 채우기
  box_padding=2
  padding_size=((2,2),(2,2))
  image_array=np.pad(image_array, padding_size, 'constant', constant_values=155)
  red_np_cropped=np.pad(red_np_cropped, padding_size, 'constant', constant_values=0)
  green_np_cropped=np.pad(green_np_cropped, padding_size, 'constant', constant_values=0)
  blue_np_cropped=np.pad(blue_np_cropped, padding_size, 'constant', constant_values=0)
  print(len(blue_np_cropped[0]))
  forward_direction=range(box_padding,len(image_array)-box_padding)
  backward_direction=range(len(image_array)-box_padding-1,box_padding+1,-1)
  downward_direction=range(box_padding,len(image_array[0])-box_padding)
  upward_direction=range(len(image_array[0])-box_padding-1,box_padding+1,-1)
  ##박스 만들기
  directions =[(-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2),(-1, -2), (-1, -1),
   (-1, 0), (-1, 1), (-1, 2),(0, -2)  ,(0, -1)  ,(0, 1)  ,(0, 2),(1, -2)  ,
   (1, -1)  ,(1, 0)  ,(1, 1)  ,(1, 2),(2, -2)  ,(2, -1)  ,(2, 0)  ,(2, 1)  ,(2, 2)]
  directions2 = [(-1, 0), (1, 0), (0, -1), (0, 1),(1,1),(-1,1),(1,-1),(-1,-1)]
  #순회하면서 검은색을 만났을때 주변을 확인하여 검은색이 아니면 합쳐서 평균값 넣어주기
  for i in forward_direction:
    for j in downward_direction:
      if image_array[i][j] == 155:
        red_sum_color=0
        green_sum_color=0
        blue_sum_color=0
        count=0
        max=0
        for x,y in directions:
          if(image_array[i+x][j+y]!=155):
            if (max < (red_np_cropped[i+x][j+y]+green_np_cropped[i+x][j+y]+blue_np_cropped[i+x][j+y])):
              max=red_np_cropped[i+x][j+y]+green_np_cropped[i+x][j+y]+blue_np_cropped[i+x][j+y]
              red_sum_color=red_np_cropped[i+x][j+y]
              green_sum_color=green_np_cropped[i+x][j+y]
              blue_sum_color=blue_np_cropped[i+x][j+y]
              image_array[i][j]=3
          red_np_cropped[i][j] =red_sum_color
          green_np_cropped[i][j] =green_sum_color
          blue_np_cropped[i][j] =blue_sum_color
        for x,y in directions:
          if(image_array[i+x][j+y]!=155):
            if (max < (red_np_cropped[i+x][j+y]+green_np_cropped[i+x][j+y]+blue_np_cropped[i+x][j+y])):
              max=red_np_cropped[i+x][j+y]+green_np_cropped[i+x][j+y]+blue_np_cropped[i+x][j+y]
              red_sum_color=red_np_cropped[i+x][j+y]
              green_sum_color=green_np_cropped[i+x][j+y]
              blue_sum_color=blue_np_cropped[i+x][j+y]
          red_np_cropped[i][j] =red_sum_color
          green_np_cropped[i][j] =green_sum_color
          blue_np_cropped[i][j] =blue_sum_color

  ## 셔플할 방향을 정해주고 셔플로직을 호출합니다
  def suffle_emptyspace_around(box_padding,direction2,num):
    left_top_started_suffle=0
    left_bottom_started_suffle=1
    right_bottom_started_suffle=2
    right_top_started_suffle=3

    if(num%4==left_top_started_suffle):
      for i in forward_direction:
          for j in downward_direction:
            suffle_emptyspace_logic(image_array,red_np_cropped,green_np_cropped,blue_np_cropped,i,j)
    if(num%4==left_bottom_started_suffle):
      for i in backward_direction:
          for j in downward_direction:
            suffle_emptyspace_logic(image_array,red_np_cropped,green_np_cropped,blue_np_cropped,i,j)

    if(num%4==right_bottom_started_suffle):
      for i in backward_direction:
          for j in upward_direction:
            suffle_emptyspace_logic(image_array,red_np_cropped,green_np_cropped,blue_np_cropped,i,j)
    if(num%4==right_top_started_suffle):
      for i in forward_direction:
          for j in upward_direction:
            suffle_emptyspace_logic(image_array,red_np_cropped,green_np_cropped,blue_np_cropped,i,j)

  ## 셔플로직
  def suffle_emptyspace_logic(image_array,red_np_cropped,green_np_cropped,blue_np_cropped,row,col):
    if image_array[row][col] == 3:
      red_sum_color=0
      green_sum_color=0
      blue_sum_color=0
      count=0
      for x,y in directions2:
        red_sum_color+=red_np_cropped[row+x][col+y]
        green_sum_color+=green_np_cropped[row+x][col+y]
        blue_sum_color+=blue_np_cropped[row+x][col+y]
        count+=1
      red_np_cropped[row][col] =red_sum_color/count
      green_np_cropped[row][col] =green_sum_color/count
      blue_np_cropped[row][col] =blue_sum_color/count


  for num in range(8):
    suffle_emptyspace_around(box_padding,directions2,num)
  # 이과정은 일단생략
  # for i in range(2,len(image_array)-2):
  #         for j in range(len(image_array[0])-3,1,-1):
  #           if image_array[i][j] == 3:
  #             for x,y in directions:
  #               if(image_array[i+x][j+y] != 3):
  #                 red_np_cropped[i+x][j+y]= red_np_cropped[i][j]
  #                 green_np_cropped[i+x][j+y]= green_np_cropped[i][j]
  #                 blue_np_cropped[i+x][j+y]=blue_np_cropped[i][j]
  for i in range(1,len(image_array)-1):
      for j in range(1,len(image_array[0])-1):
        if image_array[i][j] == 3:
          red_np_cropped[i][j] = red_np_cropped[i][j]+np.random.randint(-5,5)
          green_np_cropped[i][j] = green_np_cropped[i][j]+np.random.randint(-5,5)
          blue_np_cropped[i][j] = blue_np_cropped[i][j]+np.random.randint(-5,5)


  image = PILImage.fromarray(image_array)

  image = image.convert('L')
  red_image = PILImage.fromarray(red_np_cropped.astype(np.uint8), mode='L')
  green_image = PILImage.fromarray(green_np_cropped.astype(np.uint8), mode='L')
  blue_image = PILImage.fromarray(blue_np_cropped.astype(np.uint8), mode='L')
  mergeImage=(PILImage.merge("RGB", (red_image, green_image, blue_image)))
  mergeImage.save(f"{save_dir}/{instance_id.id}.png",format="png")
  # print(f"{save_dir}/{instance_id}.png")

@dataclass
class Instance:
  id: UUID
  mask_url: str
  label:str
  box_coordinates: List[Tuple[int, int]]

def validate_instance(instance: Instance) -> bool:
  if not isinstance(instance.id, UUID):
    return False
  if not isinstance(instance.mask_url, str):
    return False
  if not isinstance(instance.box_coordinates, list):
    return False
  if not all(isinstance(coord, tuple) and len(coord) == 2 for coord in instance.box_coordinates):
    return False
  return True

def prediction(input_img_path: str) -> List[Instance]:
  instance_info: List[Instance] = []
  img_size = (1024, 1024)
  batch_size = 8

  test_input_img_path = input_img_path
  test_gen = xBDInference(batch_size, img_size, [test_input_img_path])
  test_preds = model.predict(test_gen)
  components = find_components(np.array(test_preds[0]))
  components=list(filter(lambda x: len(x) > 100, components))
  flattened_components = [component for component in components]
  matrix_components = [find_boundaries(component) for component in components]

  ## 그림자 판단 로직

  print(len(matrix_components))
  for i in range(len(matrix_components)):
    unique_id = uuid4()
    instance = Instance(
      id=unique_id,
      mask_url=f'/static/{unique_id}.png',
      label='building',
      box_coordinates=[
         (matrix_components[i][2], matrix_components[i][3]),
         (matrix_components[i][4], matrix_components[i][3]),
         (matrix_components[i][4], matrix_components[i][5]),
         (matrix_components[i][2], matrix_components[i][5])
      ]
    )
    if not validate_instance(instance):
      instance_info = []
    else:
      instance_info.append(instance)
      save_sperate_image(unique_id, test_input_img_path, matrix_components[i], flattened_components[i])
    # print('건물 저장 완료')
  unique_id = uuid4()
  instance=Instance(
    id=unique_id,
    mask_url=f'/static/{unique_id}.png',
    label='background',
    box_coordinates=[
        (0,0),
        (0,1024),
        (1024,1024),
        (1024,0),
    ]
  )
  print(instance)

  # print('인스턴스 저장 완료')
  instance_info.append(instance)
  save_sperate_background_image(instance_info[-1], test_input_img_path, flattened_components)
  return instance_info
