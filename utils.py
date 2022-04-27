from email.mime import base
import numpy as np
from sklearn.preprocessing import minmax_scale

class SampleImage():
    def __init__(self, shape):
        self.shape = shape
        #self.pixels = np.zeros(shape=shape)
        self.sample = np.ones(shape=shape) == 1
        self.max_square_size = int(min(self.shape)/10)
        self.max_circle_radius = np.floor(np.min(self.shape)/15)
         
        self.pixels = minmax_scale(np.random.normal(size=np.prod(shape))).reshape(shape)

    def add_square(self, square_size, location):
        # obtain pixels contained in square
        idx = np.meshgrid(np.arange(location[0], location[0] + square_size).astype('int'), np.arange(location[1], location[1] + square_size).astype('int'))
        
        # set pixel values for square pixel
        self.pixels[idx[1], idx[0], 0] = 1
        self.pixels[idx[1], idx[0], 1] = 0
        self.pixels[idx[1], idx[0], 2] = 0
            
        # remove coloured pixels from sample
        self.sample[idx[1], idx[0]] = False

    def add_squares(self, n):
        for i in range(n):
            # sample square size
            square_size = np.random.choice(np.arange(4, self.max_square_size))
            
            # sample top left corner point for square
            base_point = (
                np.random.choice(np.where(self.sample == True)[0][np.where(self.sample == True)[0] < self.shape[0] - square_size]), 
                np.random.choice(np.where(self.sample == True)[1][np.where(self.sample == True)[1] < self.shape[1] - square_size])
            )
            
            self.add_square(square_size, base_point)    

    def add_circle(self, radius, location):
        # obtain indices of circle
        idx = np.where(
            (np.meshgrid(np.arange(self.shape[1]), np.arange(self.shape[0]))[1] - location[0])**2 + 
            (np.meshgrid(np.arange(self.shape[1]), np.arange(self.shape[0]))[0] - location[1])**2 < radius**2
            )
            
        # set pixel values for circle pixels
        self.pixels[idx[0], idx[1], 0] = 0
        self.pixels[idx[0], idx[1], 1] = 0
        self.pixels[idx[0], idx[1], 2] = 1
            
        # remove coloured pixels from sample
        self.sample[idx] = False

    def add_circles(self, n):
        for i in range(n):
            # sample radius for circle
            radius = np.random.choice(np.arange(4, self.max_circle_radius))
            
            # sample center point for circle
            base_point = (
                np.random.choice(np.where(self.sample == True)[0][(radius < np.where(self.sample == True)[0])*(np.where(self.sample == True)[0] < self.shape[0] - radius)]), 
                np.random.choice(np.where(self.sample == True)[1][(radius < np.where(self.sample == True)[1])*(np.where(self.sample == True)[1] < self.shape[1] - radius)])
                )
            self.add_circle(radius, base_point)
            
def generate_background(lower=10, upper=64):
    shape = np.random.choice(range(lower, upper)), np.random.choice(range(lower, upper)), 3
    return SampleImage(shape).pixels

def generate_circle(lower=10, upper=64):
    shape = np.random.choice(range(lower, upper)), np.random.choice(range(lower, upper)), 3
    radius = np.random.choice(np.arange(np.max((np.ceil(np.min(shape[0:2]) / 4), 4)), np.floor(np.min(shape[0:2]) / 2)))
    location = np.random.choice(np.arange(radius, shape[0] - radius)), np.random.choice(np.arange(radius, shape[1] - radius))
    image = SampleImage(shape)
    image.add_circle(radius, location)
    return image.pixels

def generate_square(lower=10, upper=64):
    shape = np.random.choice(range(lower, upper)), np.random.choice(range(lower, upper)), 3
    square_size = np.random.choice(np.arange(np.floor(np.min(shape[0:2])/2), np.min(shape[0:2])))
    location = np.random.choice(np.arange(shape[1] - square_size)), np.random.choice(np.arange(shape[0] - square_size))
    image = SampleImage(shape)
    image.add_square(square_size, location)
    return image.pixels
