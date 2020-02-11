import os
import numpy as np
import cv2 # OpenCV
from sklearn.svm import SVC # SVM klasifikator
from sklearn.metrics import accuracy_score
from joblib import dump, load
import functions

nbins = 9 # broj binova
cell_size = (6, 6) # broj piksela po celiji
block_size = (3, 3) # broj celija po bloku

neg ='.'+ os.path.sep+'negative'+os.path.sep
daisy ='.'+ os.path.sep+'flowers'+os.path.sep+'daisy'+os.path.sep
rose ='.'+ os.path.sep+'flowers'+os.path.sep+'rose'+os.path.sep
sunflower ='.'+ os.path.sep+'flowers'+os.path.sep+'sunflower'+os.path.sep


model = load( 'flowers.joblib')
if model == None:
    train_X =[]
    labels=[]
    for glavni in os.listdir(daisy):
            img = cv2.cvtColor(cv2.imread(daisy+glavni), cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (80, 80), interpolation = cv2.INTER_AREA)
            hog = cv2.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1], 
                                      img.shape[0] // cell_size[0] * cell_size[0]),
                            _blockSize=(block_size[1] * cell_size[1],
                                        block_size[0] * cell_size[0]),
                            _blockStride=(cell_size[1], cell_size[0]),
                            _cellSize=(cell_size[1], cell_size[0]),
                            _nbins=nbins)
            train_X.append(hog.compute(img))
            labels.append('d')
    for glavni in os.listdir(rose):
            img = cv2.cvtColor(cv2.imread(rose+glavni), cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (80, 80), interpolation = cv2.INTER_AREA)
            hog = cv2.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1], 
                                      img.shape[0] // cell_size[0] * cell_size[0]),
                            _blockSize=(block_size[1] * cell_size[1],
                                        block_size[0] * cell_size[0]),
                            _blockStride=(cell_size[1], cell_size[0]),
                            _cellSize=(cell_size[1], cell_size[0]),
                            _nbins=nbins)
            train_X.append(hog.compute(img))
            labels.append('r')
    for glavni in os.listdir(sunflower):
            img = cv2.cvtColor(cv2.imread(sunflower+glavni), cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (80, 80), interpolation = cv2.INTER_AREA)
            hog = cv2.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1], 
                                      img.shape[0] // cell_size[0] * cell_size[0]),
                            _blockSize=(block_size[1] * cell_size[1],
                                        block_size[0] * cell_size[0]),
                            _blockStride=(cell_size[1], cell_size[0]),
                            _cellSize=(cell_size[1], cell_size[0]),
                            _nbins=nbins)
            train_X.append(hog.compute(img))
            labels.append('s')
            
    x = np.array(train_X)
    y = np.array(labels)
    x_train = functions.reshape_data(x)
    print('Train shape: ', x.shape,y.shape)
    clf_svm = SVC(kernel='linear') 
    clf_svm.fit(x_train, y)
    y_train_pred = clf_svm.predict(x_train)
    print("Train accuracy: ", accuracy_score(y, y_train_pred))
    dump(clf_svm, 'flowers.joblib')
    
result = functions.testImg('suns.jpg',model)
if result == 'd':
    print('daisy')
elif result == 'r':
    print('rose')
else:
    print('sunflower')
'''
cap = cv2.VideoCapture('sinV.mp4')
    while(True):   
            ret, frame = cap.read()
            img = image_gray(frame)
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
            #x_train = reshape_data(x)
            print(model.predict(x_train)[0])
            if cv2.waitKey(1) == 27:
                break
    cap.release()
'''

