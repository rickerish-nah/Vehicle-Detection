# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

## Aim
The goal of this project is to write a software pipeline to detect vehicles in a video. The steps of this project are the following:

* Perform Histogram of Oriented Gradients (HOG) feature extraction on the labeled training dataset of images. We can also add the the histogram of color and spatial binned features to the above HOG features.

* Train the classifier on the extracted features.

* Implement a sliding-window technique and use the trained classifier to detect the cars.

* NOTE: Use/Create heat maps of recurring detections frame by frame to reject outliers and follow detected vehicles.

"""and use your trained classifier to search for vehicles in images.


* Architect a pipeline to perform the above operation and run video stream.

## Dataset
Here are links to the labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples to train your classifier.  These example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself.   You are welcome and encouraged to take advantage of the recently released [Udacity labeled dataset](https://github.com/udacity/self-driving-car/tree/master/annotations) to augment your training data.  

Some example images for testing your pipeline on single frames are located in the `test_images` folder.  To help the reviewer examine your work, please save examples of the output from each stage of your pipeline in the folder called `ouput_images`, and include them in your writeup for the project by describing what each image shows.    The video called `project_video.mp4` is the video your pipeline should work well on. 
Sample car and non car images are shown below.

![png](output_2_0.png)



![png](output_2_1.png)



![png](output_2_2.png)



![png](output_2_3.png)

## Feature Visualization
### Binned features of the color image

![png](output_7_1.png)

### Color histogram features

![png](output_8_1.png)


### HOG features

![png](output_9_1.png)

## Feature extraction and Training

All the three namely HOG, spatial binned and color histogram features were extracted. The brief detail on how many and how long are shown below.

```

    100%|██████████| 8792/8792 [00:52<00:00, 167.83it/s]
    100%|██████████| 8968/8968 [00:54<00:00, 163.43it/s]


    Using: 15 orientations 8 pixels per cell and 2 cells per block
    Time taken for feature extraction: 167.73708081245422
    Feature vector length: 11988



```python
The hyperparameters selected are,

|Parameter|Value|
|:--------|----:|
|Color Space|YUV|
|HOG Orient|15|
|HOG Pixels per cell|8|
|HOG Cell per block|2|
|HOG Channels|All|
|Spatial bin size| (32,32)|
|Histogram bins|32|
|Histogram range|(0,256)|
|Classifier|SVC(kernel = linear)|
|Scaler|StandardScaler|

These parameters were chosen manually by trial and error method. After so many trial I settled on the above mentioned parameters because they showed a good training accuracy on the model. A `LinearSVC` was used as the classifier for this project. The features are extracted and concatenated. It is very important to normalize the extracted features and randomize the entire data set to split it as Training and Validation set. Normalizing ensures that a classifier's behavior isn't dominated by just a subset of the features, and that the training process is as efficient as possible. The time taken to train and predict are shown below.
```

    28.93 Seconds to train SVC...
    Test Accuracy of SVC: 99.01%
    
      Predictions: [0. 1. 0. 0. 0. 0. 1. 1. 1. 0.]
           Labels: [0. 1. 1. 0. 0. 0. 1. 1. 1. 0.]
    
    0.01589 seconds to predict 10 labels with SVC.
```python


## Sliding Window Technique

The **`slide_window`** function takes in an image, start and stop positions, window size and overlap fraction and returns a list of bounding boxes for the search windows, which will then be passed to draw boxes.
I started by inspecting the minimum and maximum X and Y values manually by observing the cars in the sample video. I then experimented with several approaches, starting from 0.5 overlap, and approaching 0.9 overlap. Using windows of multiple sizes with varying offset was helpful in finding cars that are of larger or smaller size. The performance of the method calculating HOG on each particular window was slow. To improve the processing performance, a HOG sub-sampling was implemented as suggested on Udacity's lectures.
A few examples are shown below

![png](output_16_0.png)



![png](output_16_4.png)



![png](output_16_5.png)



### Adding Heatmaps and Bounding Boxes

The **`add_heat`** function creates a map of positive "car" results found in an image by adding all the pixels found inside of search boxes. More boxes means more "hot" pixels. The **`apply_threshold`** function defines how many search boxes have to overlap for the pixels to be counted as "hot", as a result the "false-positve" search boxes can be discarded. The **`draw_labeled_bboxes`** function takes in the "hot" pixel values from the image and converts them into labels then draws bounding boxes around those labels. Below is an example of these functions at work.


![png](output_21_2.png)



![png](output_21_3.png)



![png](output_21_4.png)


### Pipeline implementation
The structure of the architected pipeline is shown below.

Image --> applying search window --> detecting cars --> heat maps to remove false-positives -->drawing boxed on detected cars.
```python
def pipeline_detect_car(image):
    #image = mpimg.imread(img)
    output_image, bboxes = apply_sliding_window(image, svc, X_scaler, pix_per_cell, cell_per_block, size, nbins)
    heatmap = get_heatmap(bboxes)
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(image), labels)
    return draw_img
```


![png](output_23_0.png)



![png](output_23_5.png)



Since the pipeline works well for images, we now take it to the next level of passing the video onto the pipeline.
It took 17 minutes for the entire video to be processed for detection and drawing the bounding boxes as shown below.
```

    [MoviePy] >>>> Building video project_output.mp4
    [MoviePy] Writing video project_output.mp4


    100%|█████████▉| 1260/1261 [17:01<00:00,  1.23it/s]


    [MoviePy] Done.
    [MoviePy] >>>> Video ready: project_output.mp4 
    
    CPU times: user 15min 11s, sys: 54.3 s, total: 16min 6s
    Wall time: 17min 2s



```python
## Discussion
The pipeline works absolutely well but still detects a few false positives, especially poor when shadows are classified as `cars`. One thing we can do is to add more dark images to the `non-vehicle` dataset. Instead of using SVMs to classify and detect cars, which is very slow, we can use CNNs.




