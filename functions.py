import numpy as np
import cv2 # OpenCV

nbins = 9 # broj binova
cell_size = (6, 6) # broj piksela po celiji
block_size = (3, 3) # broj celija po bloku

def testImg(path,model):
    train_X=[]
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (80, 80), interpolation = cv2.INTER_AREA)
    hog = cv2.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1], 
                                      img.shape[0] // cell_size[0] * cell_size[0]),
                            _blockSize=(block_size[1] * cell_size[1],
                                        block_size[0] * cell_size[0]),
                            _blockStride=(cell_size[1], cell_size[0]),
                            _cellSize=(cell_size[1], cell_size[0]),
                            _nbins=nbins)
    train_X.append(hog.compute(img))
    x = np.array(train_X)        
    x_train = reshape_data(x)
    return model.predict(x_train)[0]


def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
def reshape_data(input_data):
    nsamples, nx, ny = input_data.shape
    return input_data.reshape((nsamples, nx*ny))