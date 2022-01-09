from PIL import Image
import cv2
from PIL import ImageFont, ImageDraw, Image
import numpy as np
import matplotlib.pyplot as plt

img = Image.open('1.jpg')
plt.figure("dog",figsize=(4,4))
plt.text(100,270,'XXXXXXXX')
plt.text(100,290,'XXXXXXXX')
plt.imshow(img)
plt.axis('off')
plt.savefig('1.png')
plt.show()

