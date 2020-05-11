This is the object detector that performs the detections of traffic signs. 
The detector has the SSD-MobileNet model and the Faster R-CNN model installed. An executable file was not created due to various compatibility issues. 

To run the program, ensure that the required python libraries are installed:
- Tensorflow 
- Protobuf
- OpenCV
- PIL
- TKinter 
- PyQt
- numpy
- aenum
- six

The "traffic_sign_detector.py" file has to be executed with the command "python traffic_sign_detector.py".

There are 3 input methods: 
1) Image: The detector will load an image and detect traffic signs from the image 
2) Video: The detector will load a video and detect traffic signs from the footage 
3) Live: The detector will load the webcam of the device and detect traffic signs from the camera input

You can switch between the two detection models. 

After selecting the input type and the detection model, press 'Start Detection" to being detecting objects.

You will be prompted with a window to select an image or a video if the input type is an image or video.

If the inputs are from a video file or the live camera, use the key 'q' to close the window. 
If the inputs are from an image, simply closing the window will exit. 


