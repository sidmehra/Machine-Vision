import cv2
import glob
import numpy as np
import os
import matplotlib.pyplot as plt
import copy
from mpl_toolkits.mplot3d import Axes3D

### Task 1
## Task 1 A
def task1A(index, image, last=False):
  '''
  Displays the checkerboard pattern to sub-pixel
  accuracy for the image passed to it
  '''

  # Converts the image to GRAYSCALE
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  # Define the termination criteria for the refinement of the pixel co-ordinates
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

  # Define the shape of the checker-board
  shape = (5, 7)

  # Extract the 2-d image co-ordinates of the checker-board corners in each image
  ret, corners = cv2.findChessboardCorners(gray_image, shape, None)

  # Defines the world-coordinates for the 3-D points
  objp = np.zeros((1, shape[0] * shape[1], 3), np.float32)
  objp[0,:,:2] = np.mgrid[0:shape[0], 0:shape[1]].T.reshape(-1, 2)

  if ret:
    # Refines the coordinates to sub-pixel accuracy
    refined_corners = cv2.cornerSubPix(gray_image, corners, (11,11),(-1,-1), criteria)

    # Draws the chessboard corners
    image = cv2.drawChessboardCorners(image, shape, refined_corners, ret)
    if not last:
      print('Displaying corners for image {}.\npress [ENTER] for the next image\n'.format(index+1))
    else:
      print('Displaying corners for image {}.\npress [ENTER] to continue\n'.format(index+1))

    # Displays the chessbiard corners
    cv2.imshow('Corners for image {}'.format(index+1), image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Resolution of the image
    resolution = image.shape[0:2]

    # Save the image to the ./output dir
    cv2.imwrite('./output/Corners_{}.jpg'.format(index+1), image)

    return [objp, refined_corners, resolution]
  else:
    print('No pattern found for image {}\npress [ENTER] for the next image.'.format(index+1))

    # Displays the image when no patterns are found
    cv2.imshow('Image {}'.format(index+1), image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return ret

## Task 1 B
def task1B(object_points, image_points, image_resolution):
  """
  1. Determine and output the camera calibration matrix K
  """

  # Perform the camera calibration
  # This function returns the following
  # K----> The common intrinsic calibration matrix
  # d----> Radial distortion parameters
  # r----> Rotation vectors in Rodriguez representation
  # t----> Camera positions in checkerboard co-ordinate system (3*1 translation vector)

  ret, K, d, rvec, tvec = cv2.calibrateCamera(object_points, image_points, image_resolution, None, None)

  # Output the calibration matrix K
  print("Calibration matrix: ")
  print(np.round(K,4), '\n')
  cont = input('Press [ENTER] to continue')
  print('\n\n')
  return K, d, rvec, tvec


## Task 1 C
def task1C():
  '''
  1. Identifies the good points to track in the first frame
  2. Refines the feature point coordinates to sub-pixel accuracy
  '''

  # Define the termination criteria for the refinement of the pixel co-ordinates
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

  # Reads the video and the first frame from it
  vid = cv2.VideoCapture('Assignment_MV_02_video.mp4')
  _, frame1 = vid.read()
  gray_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

  # Detects the good features and refines it
  good_features = cv2.goodFeaturesToTrack(gray_frame1, 200, 0.3, 7)
  good_features = cv2.cornerSubPix(gray_frame1, good_features, (11,11), (-1,-1), criteria)

  # Marks the good features
  for feature in good_features:
    frame1 = cv2.circle(frame1, tuple(feature[0]), 4, (0,255,0), thickness=-1)

  # Display the image
  print('Good features to track in FRAME 1\npress [ENTER] to continue')
  cv2.imshow('Good features to track in FRAME 1', frame1)

  # Saves the output
  cv2.imwrite('./output/Good_features.jpg', frame1)

  cv2.waitKey(0)
  cv2.destroyAllWindows()


## Task 1D
def task1D():
  imgs = []
  # Reads the Input
  camera = cv2.VideoCapture('Assignment_MV_02_video.mp4')

  # Variable to store the Output
  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  out = cv2.VideoWriter('./output/FeatureTracking.mp4', fourcc, 20.0, (1000,562))

  # Define the termination criteria for the refinement of the pixel co-ordinates
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

  # initialise feature point location to start tracking 
  # get the first frame from the camera 
  while camera.isOpened():
    ret,img= camera.read()
    if ret:
      new_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)        
      p0 = cv2.goodFeaturesToTrack(new_img, 200, 0.3, 7)   
      p0 = cv2.cornerSubPix(new_img, p0, (11,11), (-1,-1), criteria)                                 
      break    

  # initialise tracks
  # track the moving indexes  
  index = np.arange(len(p0))
  tracks = {}
  for i in range(len(p0)):
    tracks[index[i]] = {0:p0[i]}

  # Initialize frames
  frame = 0
  while camera.isOpened():
    ret,img = camera.read()
    frame += 1
    if ret:
      timg = copy.deepcopy(img)
      imgs.append(timg)
      old_img = new_img
      new_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

      # calculate optical flow
      if len(p0)>0:
        p1, st, err  = cv2.calcOpticalFlowPyrLK(old_img, new_img, p0, None)

        # visualise points
        for i in range(len(st)):
          if st[i]:
            cv2.circle(img, (p1[i,0,0],p1[i,0,1]), 2, (0,0,255), 2)
            cv2.line(img, (p0[i,0,0],p0[i,0,1]), (int(p0[i,0,0]+(p1[i][0,0]-p0[i,0,0])*5),int(p0[i,0,1]+(p1[i][0,1]-p0[i,0,1])*5)), (0,0,255), 2)            

        p0 = p1[st==1].reshape(-1,1,2)
        index = index[st.flatten()==1]

      # refresh features, if too many lost
      if len(p0)<100:
        new_p0 = cv2.goodFeaturesToTrack(new_img, 200-len(p0), 0.3, 7)
        new_p0 = cv2.cornerSubPix(new_img, new_p0, (11,11), (-1,-1), criteria) 
        for i in range(len(new_p0)):
          if np.min(np.linalg.norm((p0 - new_p0[i]).reshape(len(p0),2),axis=1))>10:
            p0 = np.append(p0,new_p0[i].reshape(-1,1,2),axis=0)
            index = np.append(index,np.max(index)+1)

      # update tracks
      for i in range(len(p0)):
        if index[i] in tracks:
          tracks[index[i]][frame] = p0[i]
        else:
          tracks[index[i]] = {frame: p0[i]}

      # visualise last frames of active tracks
      for i in range(len(index)):
        for f in range(frame-20,frame):
          if (f in tracks[index[i]]) and (f+1 in tracks[index[i]]):
            cv2.line(img,
                   (tracks[index[i]][f][0,0],tracks[index[i]][f][0,1]),
                   (tracks[index[i]][f+1][0,0],tracks[index[i]][f+1][0,1]),
                   (0,255,0), 1)
            
      # save the images as video and display it
      out.write(img)
      cv2.imshow("Live Feature Tracking", img)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    else:
      camera.release()
      out.release()
      cv2.destroyWindow("Live Feature Tracking")
      print('The video has reached its last frame.')
  return tracks, frame, imgs[-2], imgs[0]


##Task 1
def task1():
  print('-------------------Task1 A-----------------------------')
  object_points = []
  image_points = []
  images = glob.glob("Assignment_MV_02_calibration/*.png")
  for index, file_name in enumerate(images):
    # Reads the image
    image = cv2.imread(file_name)
    last = index == len(images) - 1
    output = task1A(index, image, last)

    # Checks if a checkerboard pattern has been found
    if output is not False:
      object_points.append(output[0])
      image_points.append(output[1])

  print('-------------------Task1 B-----------------------------')
  K, d, rvec, tvec = task1B(object_points, image_points, output[2])

  print('-------------------Task1 C-----------------------------')
  task1C()
  print('-------------------Task1 D-----------------------------')
  tracks, frame, img, img2 = task1D()
  return tracks, frame, img, img2, K, d, rvec, tvec

### Task 2
## Task 2 A
def task2A(tracks, frame, img):
  '''
  1. Extracts the common features from the first and last frame
  2. Visualises the correspondences
  '''
  # The first and the last frames
  frame1 = 0
  frame2 = frame-1

  # Extract the common features & visualise
  correspondences = []
  first_last_corresp = {}
  for track in tracks:
    if (frame1 in tracks[track]) and (frame2 in tracks[track]):
      x1 = [tracks[track][frame1][0,0],tracks[track][frame1][0,1],1]
      x2 = [tracks[track][frame2][0,0],tracks[track][frame2][0,1],1]
      correspondences.append([np.array(x1), np.array(x2)])
      # visualise the common features
      first_last_corresp[track] = {frame1:tuple(tracks[track][frame1].astype('int32').tolist()[0]),
                                        frame2:tuple(tracks[track][frame2].astype('int32').tolist()[0])}
      cv2.line(img, 
                first_last_corresp[track][frame1],
                first_last_corresp[track][frame2],
                (0,255,0), 1)
      cv2.circle(img, first_last_corresp[track][frame1], 1, (0, 255, 255), -1)
      cv2.circle(img, first_last_corresp[track][frame2], 1, (0, 255, 255), -1)

  print('Correspondence between the first and last frames: \nPress [ENTER] to continue')
  cv2.imwrite('output/task2A.jpg', img)
  cv2.imshow('task2A', img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

  ucorrespondences = copy.deepcopy(correspondences)
  # Use Euclidean normalized vectors
  norm_correspondences = copy.deepcopy(correspondences)
  for i, c in enumerate(correspondences):
    for j, e in enumerate(c):
      norm_correspondences[i][j] = e / np.linalg.norm(e)

  correspondences = norm_correspondences
  
  return correspondences, ucorrespondences


## Task 2 B
def task2B(correspondences):
  '''
  1. mean and standard deviations in the first & the last frames
  2. normalization using homographies
  '''
  # mean feature coordinates
  x1 = list(map(list, zip(*correspondences)))[0]
  x2 = list(map(list, zip(*correspondences)))[1]
  mu_x1 = np.mean(x1, axis=0)
  mu_x2 = np.mean(x2, axis=0)
  # standard deviations
  sigma_x1 = np.std(x1, axis=0)
  sigma_x2 = np.std(x2, axis=0)
  print("Mean feature coordinates of first frame: ({}, {}, {})".format(round(mu_x1[0], 4), round(mu_x1[1], 4), round(mu_x1[2], 4)))
  print("Mean feature coordinates of last frame: ({}, {}, {})".format(round(mu_x2[0], 4), round(mu_x2[1], 4), round(mu_x2[2], 4)))
  print("Std. dev. of feature coordinates of first frame: ({}, {}, {})".format(round(sigma_x1[0], 4), round(sigma_x1[1], 4), round(sigma_x1[2], 4)))
  print("Std. dev. of feature coordinates of last frame: ({}, {}, {})".format(round(sigma_x2[0], 4), round(sigma_x2[1], 4), round(sigma_x2[2], 4)))

  # homographies
  T1 = np.matrix([[1/sigma_x1[0], 0, -mu_x1[0]/sigma_x1[0]],
                  [0, 1/sigma_x1[1], -mu_x1[1]/sigma_x1[1]],
                  [0, 0, 1]])  
  T2 = np.matrix([[1/sigma_x2[0], 0, -mu_x2[0]/sigma_x2[0]],
                  [0, 1/sigma_x2[1], -mu_x2[1]/sigma_x2[1]],
                  [0, 0, 1]]) 
  # translating and scaling with homographies
  x1n = np.array(np.zeros((len(correspondences), 3)))
  x2n = np.array(np.zeros((len(correspondences), 3)))
  for i, (x1,x2) in enumerate(correspondences):
    x1n[i] = np.matmul(T1,x1)
    x2n[i] = np.matmul(T2,x2) 
  
  return x1n, x2n, T1, T2


## Task 2 C
def calculate_fundamental_matrix(selected, x1n, x2n, T1, T2):
  A = np.zeros((0, 9))
  for i in selected:
    x1 = np.array(x1n[i]).T
    x2 = np.array(x2n[i]).T
    ai = np.kron(x1.T,x2.T)
    A = np.append(A,[ai],axis=0)
            
  U,S,V = np.linalg.svd(A)    
  F = V[8,:].reshape(3,3).T

  U,S,V = np.linalg.svd(F)
  F = np.matmul(U,np.matmul(np.diag([S[0],S[1],0]),V))
  
  F = np.matmul(T2.T, np.matmul(F, T1)) 

  return F

def task2C(correspondences, x1n, x2n, T1, T2, doprint=True):
  '''
  1. selection of 8 feature correspondences at random
  2. calculation of the fundamental matrix 
  '''
  #np.random.seed(0)
  # select 8 feature correspondences at random
  l = len(correspondences)
  selected = set()
  num_selected = len(selected)
  to_select = 8
  while num_selected < to_select:
    i = np.random.randint(0, l)
    selected.add(i)
    num_selected = len(selected)
  # calculate the fundamental matrix
  F = calculate_fundamental_matrix(selected, x1n, x2n, T1, T2)
  # check if F is singular
  if doprint:
    if round(np.linalg.det(F), 4) != 0:
      print("Error: Computed fundamental matrix is not singular")
    else:
      print("The fundamental matrix for selection {}:\n{}".format(selected, np.round(F,4)))

  return selected, F


## Task 2 D
def task2D(correspondences, selected, x1n, x2n, T1, T2, doprint=True):
  '''
  1. calculation of fundamental matrix for the selected normalized correspondences
  '''
  # translating and scaling with homographies
  y1n = np.array(np.zeros((len(correspondences), 3)))
  y2n = np.array(np.zeros((len(correspondences), 3)))
  for i, x1 in enumerate(x1n):
    y1n[i] = np.matmul(T1,x1)
  for i, x2 in enumerate(x2n):
    y2n[i] = np.matmul(T2,x2) 
  
  # calculate the fundamental matrix
  F2 = calculate_fundamental_matrix(selected, y1n, y2n, T1, T2)
  # check if F is singular
  if doprint:
    if round(np.linalg.det(F2), 4) != 0:
      print("Error: Computed fundamental matrix is not singular")
    else:
      print("The fundamental matrix for the selected normalized correspondences:\n{}".format(np.round(F2,4)))
      # Apply the normalized homographies to F2 to obtain F
      F = np.matmul(T2.T, np.matmul(F2, T1))
      if round(np.linalg.det(F), 4) != 0:
        print("Error: Computed fundamental matrix is not singular")
      else:
        print("The fundamental matrix for selection {}:\n{}".format(selected, np.round(F,4)))

  return F2


## Task 2 E
def task2E(correspondences, x1n, x2n, F, selected, doprint=True):
  '''
  1. calculation of the value of the model equation for the remaining feature correspondences
  2. calculation of the variance of the model equation for the remaining feature correspondences
  '''
  l = len(correspondences)
  indexes = set(range(0, l))  
  unselected = indexes - selected
  # point observation covariance matrix
  C = np.matrix([[1, 0, 0],
                 [0, 1, 0],
                 [0, 0, 0]])
  g = []
  var = []
  if doprint:
    print("Values and variances of the model equation for unselected feature correspondences:")
  for i in unselected:
    x1 = x1n[i]
    x2 = x2n[i]
    x1 = np.matrix(x1).T
    x2 = np.matrix(x2).T
    # value of the model equation
    g_i = np.matmul(x2.T, np.matmul(F, x1))
    g.append(g_i)
    # variance of the model equation
    var_i = np.add(np.matmul(x2.T, np.matmul(F, np.matmul(C, np.matmul(F.T, x2)))),
                   np.matmul(x1.T, np.matmul(F.T, np.matmul(C, np.matmul(F, x1)))))
    var.append(var_i)
    if doprint:
      print("\tFeature: {}\t\tValue: {}\t\tVariance:{}".format(i, round(np.array(g_i)[0][0], 4), round(np.array(var_i)[0][0], 4)))

  return g, var, unselected


## Task 2 F  
def task2F(g, var, unselected, doprint=True):
  '''
  1. For each of the remaining correspondences, determination of presence of outliers
  2. Sum of the test statistics over all inliers
  '''
  T_i = []
  is_outlier = []
  outliers = []
  threshold = 6.635
  inlier_stats = 0
  if doprint:
    print("Outlier Dectection:")
  index = 0
  for i, val in enumerate(unselected):
    gi = np.array(g[i])[0][0]
    vari = np.array(var[i])[0][0]
    ti = (gi*gi) / vari
    T_i.append(ti)
    if ((gi*gi) / vari) > threshold:
      is_outlier.append(True)
      outliers.append(index)
    else:
      is_outlier.append(False)
      inlier_stats += ti
    if doprint:
      print("\tFeature: {}\t\tOutlier: {}".format(val, is_outlier[i]))
    index += 1
  if doprint:
    print("Sum of test statistics over all inliers: {}".format(round(inlier_stats, 4)))

  return T_i, is_outlier, outliers, inlier_stats

## Task 2 G
def task2G(correspondences, x1n, x2n, T1, T2, unselected, tracks):
  '''
  1. Repeat determination of outliers as defined in Task 2 F for 10000 random selections of correspondences
  2. Fundamental matrix for the selection that yielded the least number of outliers
  '''
  num_outliers = []
  outlier_table = {}
  # the least number of outliers initialised to an impossible value
  min_outliers = len(unselected) + 1
  selection = []
  previn = []
  remove_outliers = []
  num_iterations = 10000
  print("Iterations started:")
  for x in range(num_iterations):
    if x%1000 == 0 and x != 0:
      print("\t{} iterations completed".format(x))
    selected, F = task2C(correspondences, x1n, x2n, T1, T2, doprint=False)
    g, var, unselected = task2E(correspondences, x1n, x2n, F, selected, doprint=False)
    T_i, is_outlier, outliers, inlier_stats = task2F(g, var, unselected, doprint=False)
    num = 0
    for i in is_outlier:
      if i == True:
        num+=1
    if num < min_outliers:
      min_outliers = num
      selection = selected
      previn = inlier_stats
      remove_outliers = outliers
    # tie breaking
    elif num == min_outliers:
      if inlier_stats < previn:
        selection = selected
        previn = inlier_stats
        remove_outliers = outliers

    num_outliers.append(num)
  print("\t{} iterations completed".format(num_iterations))
  # removing outliers
  num_remove = len(remove_outliers)
  for val in remove_outliers:
    tracks.pop(val)

  # calculate the fundamental matrix
  F = calculate_fundamental_matrix(selection, x1n, x2n, T1, T2)
  # check if F is singular
  if round(np.linalg.det(F), 4) != 0:
    print("Error: Computed fundamental matrix is not singular")
  else:
    print("The fundamental matrix for selection {}:\n{}".format(selection, F))

  print('\n')
  print('Number of outliers removed:{}'.format(num_remove))


## Task 2 H
def calculate_epipoles(F):
    U,S,V = np.linalg.svd(F)    
    e1 = V[2,:]

    U,S,V = np.linalg.svd(F.T)    
    e2 = V[2,:]

    return e1,e2 


def task2H(is_outlier, tracks, img, F, unselected):
  '''
  1. Indicate inliers and outlines on displayed image
  2. Calculation of epipoles
  '''
  unselected = list(unselected)
  index = 0
  firstframe = 0
  for val in is_outlier:
    if val == True:
      cv2.circle(img, tuple(tracks[unselected[index]][firstframe].astype('int32').tolist()[0]), 4, (0, 0, 255), -4)
    else:
      cv2.circle(img, tuple(tracks[unselected[index]][firstframe].astype('int32').tolist()[0]), 4, (255, 0, 0), -4)
    index += 1
  cv2.imwrite('output/Task2H.jpg', img)
  print('Red indicates outliers.')
  print('Blue indicates inliers.')
  cv2.imshow('img', img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  # epipoles
  e1, e2 = calculate_epipoles(F)
  print("Coordinates of the two epipoles:")
  print("\tE1:", e1/e1[0,2])
  print("\tE2:", e2/e2[0,2])


## Task 2
def task2(tracks, frame, img):
  print('-------------------Task2 A-----------------------------')
  correspondences, ucorrespondences = task2A(tracks, frame, img)
  print('-------------------Task2 B-----------------------------')
  (x1n, x2n, T1, T2) = task2B(correspondences)
  print('-------------------Task2 C-----------------------------')
  (selected, F) = task2C(correspondences, x1n, x2n, T1, T2)
  print('-------------------Task2 D-----------------------------')
  F2 = task2D(correspondences, selected, x1n, x2n, T1, T2)
  print('-------------------Task2 E-----------------------------')
  (g, var, unselected) = task2E(correspondences, x1n, x2n, F, selected)
  print('-------------------Task2 F-----------------------------')
  (T_i, is_outlier, outliers, inlier_stats) = task2F(g, var, unselected)
  print('-------------------Task2 G-----------------------------')
  task2G(correspondences, x1n, x2n, T1, T2, unselected, tracks)
  print('-------------------Task2 H-----------------------------')
  task2H(is_outlier, tracks, img, F, unselected)
  return correspondences, ucorrespondences, selected, unselected, F, is_outlier


### Task 3
## Task 3 A
def task3A(selected, F, K):
  '''
  1. calculation of essential matrix for the selected feature correspondences of task 2 C
  2. Verify non-zero singular values of essential matrix are identical
  3. Verify the rotation matrices of the SVD  
  '''

  # essential matrix
  E = np.matmul(K.T, np.matmul(F, K))
  if np.round(np.linalg.det(E), 4) != 0:
    print("Error: Computed essential matrix is not singular")
  else:
    # singular value decomposition
    U, S, V = np.linalg.svd(E)
    # non-zero singular values of E are identical
    l = (S[0] + S[1]) / 2.0
    S[0] = S[1] = l
    # rotation matrices of the SVD have positive determinants
    if np.linalg.det(U) < 0:
      U[:,2] *= -1
    if np.linalg.det(V) < 0:
      V[2,:] *= -1
    print("The essential matrix for selection {}:\n{}".format(selected, np.round(E,4)))
    print("Singular value decomposition of essential matrix E:")
    print("U: {},\nS: {},\nV: {}".format(np.round(U,4), np.round(S,4), np.round(V,4)))
  return E, U, S, V


## Task 3 B
def task3B(frame, U, V):
  '''
  1. Determination of the four potential combinations of rotation matrices R and 
     translation vector t between the first and the last frame
  '''
  # determination of the scale of the baseline t in meters
  cam_speed = 50        # speed of motion of camera in kmph
  fps = 30
  num_frames = frame    # number of frames shot
  beta = cam_speed * (18/5.0) * (num_frames/fps)

  # determination of combinations of R and t
  W = np.matrix([[0, -1, 0],
                 [1,  0, 0],
                 [0,  0, 1]])
  Z = np.matrix([[ 0, 1, 0],
                 [-1, 0, 0],
                 [ 0, 0, 0]])
  # R' = UWV'
  R1 = (np.matmul(U, np.matmul(W, V.T))).T
  # R' = UW'V'
  R2 = (np.matmul(U, np.matmul(W.T, V.T))).T
  # S[R' t] = beta*UZU'
  rhs1 = beta * np.matmul(U, np.matmul(Z, U.T))
  #print("beta*UZU':", rhs1)
  srt1 = np.matrix([[rhs1[2,1], rhs1[0, 2], rhs1[1,0]]])
  t1 = np.matmul(np.linalg.inv(R1.T), srt1.T)
  t2 = -t1
  # S[R' t] = -beta*UZU'
  rhs2 = -rhs1
  #print("-beta*UZU':", rhs2)
  srt2 = np.matrix([[rhs2[2,1], rhs2[0, 2], rhs2[1,0]]])
  t3 = np.matmul(np.linalg.inv(R2.T), srt2.T)
  t4 = -t3
  combinations = []
  combinations.append((R1, t1))
  combinations.append((R1, t2))
  combinations.append((R2, t3))
  combinations.append((R2, t4))
  print("The four potential combinations of R and t:")
  for i, c in enumerate(combinations):
    j = 1
    if i > 1:
      j = 2
    print("Combination {}:".format(i+1))
    print("R{}: {}".format(j, np.round(c[0], 4)))
    print("t{}: {}".format(i+1, np.round(c[1], 4)))

  return combinations


## Task 3 C 
def task3C(K, correspondences, selected, unselected, is_outlier, combinations):
  '''
  1. computation of m and m' for each inlier feature correspondence
  2. calculation of the unknown distances lambda and mu
  3. computation of the 3d coordinates of the scene points
  4. determination of which {R, t} combination is correct
  5. Discard all points which are behind either of the frames for this combination as outliers
  '''
  unselected = list(unselected)
  directions = {}
  lambda_mu = {}                        # key: combination index, value: [ (feature_idx, lambda, mu) for all inlier correspondences] 
  for i, idx in enumerate(unselected):
    if is_outlier[i] != True:           # inlier
      print("m and m' for inlier feature correspondence {}:".format(idx))
      fc = correspondences[idx]         # feature correspondence
      x1 = np.matrix(fc[0]).T
      x2 = np.matrix(fc[1]).T
      # computation of m and m'
      m1 = np.matmul(np.linalg.inv(K), x1)
      m2 = np.matmul(np.linalg.inv(K), x2)
      directions[idx] = (m1, m2)
      print("\tm:{}".format(np.round(np.array(m1.T)[0],4)))
      print("\tm':{}".format(np.round(np.array(m2.T)[0],4)))
      print("lambda and mu for inlier feature correspondence {}:".format(idx)) 
      for j, c in enumerate(combinations):
        r = 1
        if j > 1:
          r = 2
        R = c[0]
        t = c[1]
        if j not in lambda_mu.keys():
          lambda_mu[j] = []
        # calculate lambda and mu
        lhs_mat = np.matrix([np.zeros(2), np.zeros(2)])
        lhs_mat[0,0] = np.matmul(m1.T, m1)
        lhs_mat[0,1] = -np.matmul(m1.T, np.matmul(R, m2))
        lhs_mat[1,0] = np.matmul(m1.T, np.matmul(R, m2))
        lhs_mat[1,1] = -np.matmul(m2.T, m2)
        rhs_mat = np.matrix([np.zeros(1), np.zeros(1)])
        rhs_mat[0,0] = np.matmul(t.T, m1)
        rhs_mat[1,0] = np.matmul(t.T, np.matmul(R, m2))
        l_m = np.matmul(np.linalg.inv(lhs_mat), rhs_mat)
        lambda_mu[j].append((idx, l_m[0,0], l_m[1,0]))
        print("\tcorresponding to (R{}, t{}):".format(r, j+1))
        print("\t\tlambda: {}".format(np.round(l_m[0,0], 4)))
        print("\t\tmu: {}".format(np.round(l_m[1,0], 4)))
  # determine which {R, t} is correct
  pos_lm_count = []
  for i in lambda_mu.keys():
    count = 0
    val = lambda_mu[i]
    for (_, l, m) in val:
      if l > 0 and m > 0:
        count+=1
    pos_lm_count.append(count)
  Rt_best = pos_lm_count.index(max(pos_lm_count))
  R, t = combinations[Rt_best]
  print("The correct solution (R, t):")
  print("\tR:")
  print(np.round(R,4))
  print("\tt:")
  print(np.round(t,4))
  # Discard all points which are behind either of the frames for this combination as outliers
  f_inliers = []
  f_outliers = []
  val = lambda_mu[Rt_best]
  print("Points discarded as outliers:")
  for (idx, l, m) in val:
    point = correspondences[idx]
    if l > 0 and m > 0:
      f_inliers.append(point)
    else:
      f_outliers.append(point)
      print("\t<{},{}>".format(np.round(point[0], 4), np.round(point[1], 4)))

  # get the 3d points
  inliers_3d = []
  for (idx, l, m) in val:
    m1, m2 = directions[idx]
    X_lambda = l * m1
    X_mu = t + m * np.matmul(R, m2)
    X = (X_lambda + X_mu) / 2
    inliers_3d.append(np.array(X))

  return val, R, t, inliers_3d


## Task 3 D 
def task3D(t, inliers_3D):
  '''
  1. 3d plot to show the two camera centres and all 3d points
  '''
  # Initialise the plot
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  camera_centre1 = [0, 0, 0]
  camera_centre2 = t
  # Visualises the camera centers
  ax.scatter(camera_centre1[0], camera_centre1[1], camera_centre1[2], c='r')
  ax.scatter(camera_centre2[0], camera_centre2[1], camera_centre2[2], c='r')
  # Visualise the 3D points
  for point in inliers_3D:
    x_1, y_1, z_1 = point
    ax.scatter(x_1, y_1, z_1, c='b')
  # Show the plot
  print('Plot of the 3D points and the 2 camera centers')
  print('\tRed represents the camera centers')
  print('\tBlue represents the 3D points')
  plt.savefig('output/plot_3D.png')
  plt.show()


## Task 3 E
def task3E(K, d, rvec, tvec, inliers_3d, val, R, t, firstframe_img, lastframe_img, ucorrespondences):
  '''
  1. projection of the 3d points into the first and last frames
  2. display their position in relation to the corresponding features
  '''
  first_pp = []
  last_pp = []
  for i, frame1_c in enumerate(inliers_3d):
    # 3D -> 2D
    arr = np.float64(frame1_c)
    x,y=cv2.projectPoints(arr, np.zeros(3), np.zeros(3), K, d)
    cv2.circle(firstframe_img, (int(x.tolist()[0][0][0]), int(x.tolist()[0][0][1])), 2, (0,255,0), -2)
    cv2.circle(firstframe_img, (int(ucorrespondences[val[i][0]][0][0]), int(ucorrespondences[val[i][0]][0][1])), 2, (255,0,0), -2)
    cv2.line(firstframe_img, (int(x.tolist()[0][0][0]), int(x.tolist()[0][0][1])),
              (int(ucorrespondences[val[i][0]][0][0]), int(ucorrespondences[val[i][0]][0][1])), (0,0,255), 1)
    cv2.circle(lastframe_img, (int(x.tolist()[0][0][0]), int(x.tolist()[0][0][1])), 2, (0,255,0), -2)
    cv2.circle(lastframe_img, (int(ucorrespondences[val[i][0]][1][0]), int(ucorrespondences[val[i][0]][1][1])), 2, (255,0,0), -2)
    cv2.line(lastframe_img, (int(x.tolist()[0][0][0]), int(x.tolist()[0][0][1])),
              (int(ucorrespondences[val[i][0]][1][0]), int(ucorrespondences[val[i][0]][1][1])), (0,0,255), 1)
  cv2.imwrite('output/plot1vscorrespondencies.jpg', firstframe_img)
  cv2.imshow('plot1vscorrespondencies', firstframe_img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  cv2.imwrite('output/plot2vscorrespondencies.jpg', lastframe_img)
  print('Plot of the 3D points and the corresponding feature points')
  print('\tGreen represents the 3D points')
  print('\tBlue represents the correspondence points')
  cv2.imshow('plot2vscorrespondencies', lastframe_img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()



## Task 3 
def task3(K, d, rvec, tvec, frame, correspondences, ucorrespondences, selected, unselected, F, is_outlier, nimg, nimg2):
  print('-------------------Task3 A-----------------------------')
  (E, U, S, V) = task3A(selected, F, K)
  print('-------------------Task3 B-----------------------------')
  combinations = task3B(frame, U, V)
  print('-------------------Task3 C-----------------------------')
  val, R, t, inliers_3d = task3C(K, correspondences, selected, unselected, is_outlier, combinations)
  print('-------------------Task3 D-----------------------------')
  task3D(t, inliers_3d)
  print('-------------------Task3 E-----------------------------')
  task3E(K, d, rvec, tvec, inliers_3d, val, R, t, nimg, nimg2, ucorrespondences)

### MAIN
def main():
  # Checks if the output directory already exists else creates it
  if not os.path.exists('output'):
    os.makedirs('output')
  (tracks, frame, img, img2, K, d, rvec, tvec) = task1()
  nimg = copy.deepcopy(img)
  nimg2 = copy.deepcopy(img2)
  (correspondences, ucorrespondences, selected, unselected, F, is_outlier) = task2(tracks, frame, img)
  task3(K, d, rvec, tvec, frame, correspondences, ucorrespondences, selected, unselected, F, is_outlier, nimg, nimg2)


main()