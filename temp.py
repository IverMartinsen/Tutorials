from utils import SampleImage
import numpy as np
import matplotlib.pyplot as plt
from cv2 import rectangle

test = SampleImage(size=(128, 128), num_circles=10, num_squares=10)

print(test.objects)
plt.imshow(rectangle(test, (0, 0), (10, 10), 3))
plt.show()