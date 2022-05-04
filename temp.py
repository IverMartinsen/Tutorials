from utils import SampleImage
import numpy as np
import matplotlib.pyplot as plt

test = SampleImage(shape=(128, 128, 3), num_circles=10, num_squares=10)


plt.imshow(test)
plt.show()