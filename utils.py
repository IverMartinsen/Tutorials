import numpy as np
from sklearn.preprocessing import minmax_scale

class SampleImage(np.ndarray):
    """
    Image of objects (circles and squares) as a numpy array. 
    Location of objects is passed as an additional attribute.    
    """
    def __new__(cls, size, num_circles, num_squares):
        canvas = Canvas(size=size)
        canvas.draw_circles(num_circles)
        canvas.draw_squares(num_squares)
        obj = np.asarray(canvas.pixels).view(cls)

        obj.objects = canvas.objects

        return obj

    def __array_finalize(self, obj):
        if obj is None: return
        self.objects = getattr(obj, 'objects')

class Canvas:
    """
    Canvas class with methods for drawing objects.
    """
    def __init__(self, size):
        super().__init__()

        self.shape = size + (3, )
        self.sample = np.ones(shape=self.shape) == 1
        self.max_square_size = int(min(size)/10)
        self.max_circle_radius = np.floor(np.min(size)/15)
        self.objects = {'circles':[], 'squares':[]}
        self.pixels = minmax_scale(np.random.normal(size=np.prod(self.shape))).reshape(self.shape)
        

    def draw_square(self, square_size, location):
        # obtain pixels contained in square
        idx = np.meshgrid(np.arange(location[1], location[1] + square_size).astype('int'), np.arange(location[0], location[0] + square_size).astype('int'))
        
        # set pixel values for square pixel
        self.pixels[idx[1], idx[0], 0] = 1
        self.pixels[idx[1], idx[0], 1] = 0
        self.pixels[idx[1], idx[0], 2] = 0
            
        # remove coloured pixels from sample
        self.sample[idx[1], idx[0]] = False

    def draw_squares(self, n):
        for i in range(n):
            # sample square size
            square_size = np.random.choice(np.arange(4, self.max_square_size))
            
            # sample top left corner point for square
            location = (
                np.random.choice(np.where(self.sample == True)[0][np.where(self.sample == True)[0] < self.shape[0] - square_size]), 
                np.random.choice(np.where(self.sample == True)[1][np.where(self.sample == True)[1] < self.shape[1] - square_size])
            )
            
            
            self.objects['squares'].append((location, (location[0] + square_size, location[1] + square_size)))

            self.draw_square(square_size, location)    

    def draw_circle(self, radius, location):
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

    def draw_circles(self, n):
        for i in range(n):
            # sample radius for circle
            radius = np.random.choice(np.arange(4, self.max_circle_radius))
            
            # sample center point for circle
            location = (
                np.random.choice(np.where(self.sample == True)[0][(radius < np.where(self.sample == True)[0])*(np.where(self.sample == True)[0] < self.shape[0] - radius)]), 
                np.random.choice(np.where(self.sample == True)[1][(radius < np.where(self.sample == True)[1])*(np.where(self.sample == True)[1] < self.shape[1] - radius)])
                )
            
            p1 = location[0] - radius, location[1] - radius 
            p2 = location[0] + radius, location[1] + radius
            self.objects['circles'].append((p1, p2))
            self.draw_circle(radius, location)

            
def generate_background(lower=10, upper=64):
    size = np.random.choice(range(lower, upper)), np.random.choice(range(lower, upper))
    return Canvas(size).pixels

def generate_circle(lower=10, upper=64):
    # sample shape (height, width, channel)
    shape = np.random.choice(range(lower, upper)), np.random.choice(range(lower, upper)), 3
    radius = np.random.choice(np.arange(np.max((np.ceil(np.min(shape[0:2]) / 4), 4)), np.floor(np.min(shape[0:2]) / 2)))
    # sample  location (row, column)
    location = np.random.choice(np.arange(radius, shape[0] - radius)), np.random.choice(np.arange(radius, shape[1] - radius))
    image = SampleImage(shape)
    image.add_circle(radius, location)
    return image.pixels

def generate_square(lower=10, upper=64):
    # sample shape (height, width, channels)
    shape = np.random.choice(range(lower, upper)), np.random.choice(range(lower, upper)), 3
    # sample square size
    square_size = np.random.choice(np.arange(np.floor(np.min(shape[0:2])/2), np.min(shape[0:2])))
    # sample location (row, column)
    location = np.random.choice(np.arange(shape[0] - square_size)), np.random.choice(np.arange(shape[1] - square_size))
    image = SampleImage(shape)
    image.add_square(square_size, location)
    return image.pixels
