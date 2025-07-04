# Object Detection and Distance Measurement

## Introduction
 This repo contains object_detection.py, which can perform the following task -
 - Object detection from a live video frame, in any video file, or in an image
 - Counting the number of objects in a frame
 - Measuring the distance of an object using depth information
 - Inference on Multiple Camera feed at a time
 
For object detection, YOLO-V3 has been used, which can detect 80 different objects. Some of those are-
- person
- car
- bus
- stop sign
- bench
- dog
- bear
- backpack, and so on.

### Installation
``` python
    $ pip install -r requirements.txt
         or
    $ pip install opencv-python
    $ pip install numpy
    $ pip install pandas
    $ pip install matplotlib
    $ pip install Pillow
    $ pip install imutils
```
<hr>

#### For the installation of torch using "pip" 
``` python
    $ pip3 install torch===1.2.0 torchvision===0.4.0 -f https://download.pytorch.org/whl/torch_stable.html
```
or please follow the instructions from [Pytorch](https://pytorch.org/)
#### For installing the "win32com.client" which is Text-to-Speech module for windows you have follow this
First, open the cmd as an administrator, then run
``` python
   $ python -m pip install pywin32
   #After installing, open your Python shell and run
      import win32com.client
      speaker = win32com.client.Dispatch("SAPI.SpVoice")
      speaker.Speak("Good Morning")
```

For multiple camera support, need to add a few lines of codes as follows in app.py-

``` python
   def simulate(camera):
       while True:
           frame = camera.main()
           if frame != "":
               yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

   @app.route('/video_simulate')
   def video_simulate():
       id = 0
       return Response(gen(ObjectDetection(id)), mimetype='multipart/x-mixed-replace; boundary=frame')
```

#### Theory
There are two well-known strategies in a traditional image classification approach for object detection.

There are two scenarios for a single object in an image.
- Classification
- Localization

There are two scenarios for multiple objects in an image.
- Object detection and localization
- Object segmentation

## Distance Measurement
Traditionally, we measure the distance of any object using Ultrasonic sensors such as HC-sr04 or any other high-frequency devices that generate sound waves to calculate the distance it traverses.
However, when you are working with an embedded device to make a compact design that has functionalities such as 

- Object detection (with camera) and 
- Distance measurement 

You don't always want to make your device heavier by adding unnecessary hardware modules. To avoid such cases, you can follow a more convenient and feasible approach. As you have already integrated a camera for object detection, you can use the depth information that the camera uses to draw the bounding boxes for localizing objects to calculate the distance of that object from the camera.

### How the object detection works?
From the initial part, we understood that to measure the distance from an image, we had to localize it first to get the depth information.
<b> Now, how localization works?</b>

#### Localize objects with regression
   Regression is about returning a number instead of a class. The number can be represented as (x0,y0,width,height) which are related to a bounding box. In the images illustrated above for single object if you want to only classify  the object type then we don't need to draw the bounding box around that object that's why this part is known as <b> Classification </b>.
   However, if we are interested to know where does this object locates in the image, then we need to know that 4 numbers that a regreesion layer will return. As you can see there is a black rectangle shape box in the image of a white dog, which was drawn using the regression layer. What happens here is that after the final convolutional layer + Fully connected layers, instead of asking for class scores to compare with some offsets, a regression layer is introduced. Regression layer is nothing but some rectangular box which represents individual objects. For every frame/image to detect objects the following things happens.
 - Using the inference on any pre-trained imagenet model the last fully connected layer will need to be re-trained to the desired objects. 
 - After that all the proposals (=~2000proposal/image) will be resized to maatch the inputs of the cnn.
 - A SVM is need to be trained to classify between object and background (One binary SVM(Support Vector Machine) for each class)
 - And to put the bounding box perfectly over the image a linear regression classifier is needed to be trained which will output some correction factor.
Problem with this approch is that one part of the network is dedicated for region proposals. After the full connected layers the model tries to propose certain regions on that image which may contain object/objects. So it also requires a high qulaity classifier to filter out valid proposals which will definitely contains object/objects. Although these methos is very accurate but it comes with a big computational cost (low frame-rate) and that's why it is not suitable for embedded devices such as Arduino or Raspberry Pi which has less processing power.
<hr>

#### Localizing with Convolution neural networks

Another way of doing object detection and to reduce this tedious work is by combining the previous two task into one network. Here, instead of proposing regions for every images the model is fed with a set of pre-defined boxes to look for objects. So prior to the training phase of a neural network some pre-defined rectangular boxes that represents some objects are given to the network to train with. So when a image is gone through the network, after the fully connected layer the trained model tries to match predefined boxes to objects on that image by using non-maxima suppression algorithm to completely tied. If the comparison crosses some threshold, the model tries to draw the bounding box over the object. For example, in the case of the picture of white dog, the model knows what is the coordinates of the box of the dog object and when the image classification is done the model uses L2 distance to calculate the loss between the actual box coordinates that was predefined and the coordinate that the model gave so that it can perfectly draw the bounding box over the object on that image.

The main idea is to use the convolutional feature maps from the later layers of a network to run small CONV filters over these feature maps to predict class scores and bounding box offsets.
Here, we are reusing the computation already made during classification to localize objects to grab the activation from the final conv layers. At this point, we still have the spatial information of an image that model starts training with but is represented in a much smaller scope. So, in the final layers, each "pixel" represent a larger area of the input image, so we can use those cells to infer object position. Here the tensor containing the original image's information is quite deep as it is now squeezed to a lower dimension. At this point, a 1x1 CONV layer can be used to classify each cell as a class, and also, from the same layer, we can add another CPNV or FC(Fully Connected) layer to predict four numbers( Bounding Box). In this way, we get both class scores and location from one. This approach is known as <b> Single Shot Detection </b>. The overall strategy in this approach can be summarised as follows:-
- Train a CNN with regression(bounding box) and classification objective.
- Gather Activation from a particular layer or  layers to infer classification and location with FC layer or another CONV layer that works like an FC layer.
- During prediction, use algorithms like non-maxima suppression to filter multiple boxes around the same object.
- During training time, use algorithms like IoU to relate the predictions during training to the ground truth.

follows the strategy of Single Shot Detection. It uses a single activation map for the prediction of classes and bounding boxes at a time that's why it is called "You Only Look Once".

Here pre-trained <b> yolo-v3 </b> has been used, which can detect <b>80 different objects</b>. Although this model is faster but it doesn't give the reliability of predicting the actual object in a given frame/image. It's a kind of trade-off between accuracy and precision.

### How the distance measurement works?
This formula is used to determine the distance 

``` python
    distancei = (2 x 3.14 x 180) ÷ (w + h x 360) x 1000 + 3
```

You can relate this image to the white dog picture where the dog was localized. Again we will get four numbers in the bounding box which is (x0,y0,width,height). Here x0,y0 is used to tiled or adjust the bounding box. Width and Height these two variables are used in the formula of measuring the object and describing the detail of the detected object/objects. Width and Height will vary depending on the distance of the object from the camera.

As we know, an image goes refracted when it goes through a lens because the ray of light can also enter the lens, whereas, in the case of a mirror, the light can be reflected. That's why we get an exact reflection of the image. But in the case of the lens image gets a little stretched. The following image illustrates how the image and the corresponding angles look when it enters through a lens.

If we see there are three variables named:

- do (Distance of object from the lens)
- di (Distance of the refracted image from the convex lens)
- f (focal length or focal distance)

So the green line <b>"do"</b> represents the actual distance of the object from the convex length. And <b>"di"</b> gives a sense of what the actual image looks like. Now if we consider a triangle on the left side of the image(new refracted image) with base <b> "do" </b> and draw an opposite triangle similar to the left side one. So the new base of the opposite triangle will also be done with the same perpendicular distance. Now if we compare the two triangles from the right side, we will see <b> "do"</b> and <b> "di" </b> is parallel, and the angle creates on each side of both triangles are opposite to each other. From this, we can infer that both the triangles on the right side are also similar. Now, as they are similar, the ratio of the corresponding sides will be also similar. So do/di = A/B. Again if we compare two triangles on the right side of the image where opposite angles are equal and one angle of both the triangles are right angle (90°) (dark blue area). So A:B is both hypotenuses of a similar triangle where both triangles has a right angle.
we will get our final result in "inches" from this formula of distance. 

``` python
    distancei = (2 x 3.14 x 180) ÷ (w + h x 360) x 1000 + 3
```

* Notes - As mentioned earlier YOLO prefers performance over accuracy that's why the model predicts wrong objects frequently.

### Reference

- [Real-Time Distance Measurement Using a Modified
Camera ](https://sci-hub.tw/10.1109/SAS.2010.5439423)
- [Real-time Distance Measurement Using Single Image](http://emaraic.com/blog/distance-measurement)
- [Object image and focal distance relationship (proof of formula)](https://www.khanacademy.org/science/physics/geometric-optics/lenses/v/object-image-and-focal-distance-relationship-proof-of-formula)
- [Distance or arc length from angular displacement](https://www.khanacademy.org/science/ap-physics-1/ap-centripetal-force-and-gravitation/introduction-to-uniform-circular-motion-ap/v/distance-or-arc-length-from-angular-displacement)
