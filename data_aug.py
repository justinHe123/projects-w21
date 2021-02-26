import torch
from torchvision import transforms.functional as tf
from PIL import Image
import opencv as cv2

def rotate(img, angle):
  return tf.rotate(image, angle)

data = np.genfromtxt(path, delimiter=',', dtype='str')
truth = data[1:int(count*train_prop), 1]
images = data[1:int(count*train_prop), 0]

for i in range(0,10):

# tensor  = tensor.cpu().numpy() # make sure tensor is on cpu
cv2.imwrite(tensor, "image.png")