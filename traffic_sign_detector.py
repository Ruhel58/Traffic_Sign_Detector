import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import tkinter as tk
from tkinter import filedialog
from utils import label_map_util
from utils import visualization_utils as vis_util
from aenum import Enum

class InputMode(Enum):
    IMAGE = 1
    VIDEO = 2
    LIVE = 3

class Model(Enum):
    _init_ = 'value string'

    SSD_MOBILENET = 1, 'ssd_detector'
    FASTER_RCNN = 2, 'rcnn_detector'

    def __str__(self):
        return self.string

class ObjectDetector:
    CWD_PATH = os.getcwd()
    PATH_TO_LABELS = os.path.join(CWD_PATH,'data','label_map.pbtxt')
    NUM_CLASSES = 25

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    def __init__(self, model, inputType):
        self.MODEL = Model(model)
        self.MODEL_NAME = str(self.MODEL)

        self.INPUT_TYPE = InputMode(inputType)
        self.PATH_TO_CKPT = os.path.join(self.CWD_PATH,self.MODEL_NAME,'frozen_inference_graph.pb')
      
    def startDetection(self):
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(self.PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            sess = tf.compat.v1.Session(graph=detection_graph)

        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        if self.INPUT_TYPE == InputMode.LIVE:
            detection_input = cv2.VideoCapture(0)
        elif self.INPUT_TYPE == InputMode.IMAGE: 
            PATH_TO_INPUT = filedialog.askopenfilename(filetypes=[("Images", "*.png"), ("Images", "*.jpg"), ("Images", "*.jpeg")])
            detection_input = cv2.VideoCapture(PATH_TO_INPUT)
        elif self.INPUT_TYPE == InputMode.VIDEO: 
            PATH_TO_INPUT = filedialog.askopenfilename(filetypes=[("Videos", "*.mov"), ("Videos", "*.avi"), ("Videos", "*.mp4")])
            detection_input = cv2.VideoCapture(PATH_TO_INPUT)

        while(detection_input.isOpened()):
            ret, frame = detection_input.read()

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_expanded = np.expand_dims(frame_rgb, axis=0)
            
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: frame_expanded})

            vis_util.visualize_boxes_and_labels_on_image_array(
                frame,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                self.category_index,
                use_normalized_coordinates=True,
                line_thickness=4,
                min_score_thresh=0.55)

            cv2.namedWindow("Object Detector", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Object Detector", (1280,720))
            cv2.imshow("Object Detector", frame)

            if self.INPUT_TYPE == InputMode.IMAGE:
                return
            else:            
                if cv2.waitKey(1) == ord('q'):
                    break
        detection_input.release()
        cv2.destroyAllWindows()
                
top = tk.Tk()

top.wm_title("Traffic Sign Detector")

option = tk.IntVar()
model = tk.IntVar()

def start():
    detector = ObjectDetector(model = model.get(), inputType = option.get())
    detector.startDetection()

title = tk.Label(top, text="Traffic Sign Detector", font=("Arial", 20))
title.grid(row =0, column=0, padx=(20,20), pady=(2,2), sticky=tk.W)

title = tk.Label(top, text="Input Type:", font=("Arial", 14))
title.grid(row =1, column=0, padx=(20,20), pady=(2,2), sticky=tk.W)

title = tk.Label(top, text="Model Type:", font=("Arial", 14))
title.grid(row =1, column=1, padx=(20,20), pady=(10,10), sticky=tk.W)

O1 = tk.Radiobutton(top, text="Detect from an image", variable=option, value=InputMode.IMAGE.value)
O1.grid(row =2, column=0, padx=(20,20), pady=(10,10), sticky=tk.W)
O1.invoke()

O2 = tk.Radiobutton(top, text="Detect from a video", variable=option, value=InputMode.VIDEO.value)
O2.grid(row =3, column=0, padx=(20,20), pady=(10,10), sticky=tk.W)

O3 = tk.Radiobutton(top, text="Detect from a live camera", variable=option, value=InputMode.LIVE.value)
O3.grid(row =4, column=0, padx=(20,20), pady=(10,10), sticky=tk.W)

M1 = tk.Radiobutton(top, text="SSD-MobileNet", variable=model, value=Model.SSD_MOBILENET.value)
M1.grid(row =2, column=1, padx=(20,20), pady=(10,10), sticky=tk.W)
M1.invoke()

M2 = tk.Radiobutton(top, text="Faster R-CNN", variable=model, value=Model.FASTER_RCNN.value)
M2.grid(row =3, column=1, padx=(20,20), pady=(10,10),sticky=tk.W)

start_button = tk.Button(text ="Start Detection", command = start)
start_button.grid(row=4, column=1, padx=(20,20), pady=(10,10), sticky=tk.E+tk.S)

top.geometry("400x210")
top.resizable(width = False, height = False)

top.mainloop()