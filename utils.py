import numpy as np

class SampleImage():
    def __init__(self, shape):
        self.shape = shape
        self.pixels = np.zeros(shape=shape)
        self.sample = np.ones(shape=shape) == 1
        self.max_square_size = int(min(self.shape)/10)
        self.max_circle_radius = np.floor(np.min(self.shape)/15)

    def add_squares(self, n):
        for i in range(n):
            # sample square size
            square_size = np.random.choice(np.arange(4, self.max_square_size))
            
            # sample top left corner point for square
            base_point = (
                np.random.choice(np.where(self.sample == True)[0][np.where(self.sample == True)[0] < self.shape[0] - square_size]), 
                np.random.choice(np.where(self.sample == True)[1][np.where(self.sample == True)[1] < self.shape[1] - square_size])
            )
            
            # obtain pixels contained in square
            idx = tuple(np.meshgrid(np.arange(base_point[0], base_point[0] + square_size), np.arange(base_point[1], base_point[1] + square_size)))
            
            # set pixel values for square pixel
            self.pixels[idx] = 0.8
            
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

            # obtain indices of circle
            idx = np.where((np.meshgrid(tuple(range(128)), tuple(range(128)))[0] - base_point[0])**2 + (np.meshgrid(tuple(range(128)), tuple(range(128)))[1] - base_point[1])**2 < radius**2)
            
            # set pixel values for circle pixels
            self.pixels[idx] = 0.5
            
            # remove coloured pixels from sample
            self.sample[idx] = False
