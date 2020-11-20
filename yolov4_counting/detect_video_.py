import os
# comment out below line to enable tensorflow outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from core.functions import *
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/video.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
# flags.DEFINE_float('iou', 0.45, 'iou threshold')  # ORIGINAL 
# flags.DEFINE_float('score', 0.30, 'score threshold') #ORIGINAL
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('count', False, 'count objects within video')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'print info on detections')
flags.DEFINE_boolean('crop', False, 'crop detections from images')
flags.DEFINE_boolean('plate', False, 'perform license plate recognition')

#bobur add
flags.DEFINE_boolean('distance', False, 'compute the distance')



def main(_argv):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video
    # get video name by using split method
    video_name = video_path.split('/')[-1]
    video_name = video_name.split('.')[0]
    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    out = None

    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    frame_num = 0
    while True:
        return_value, img = vid.read()
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        external_poly = [np.array([[0,0],[486,638],[720,642],[720,0]]),
                        np.array([[624,640],[687,802],[720,802],[720,642]]),
                        np.array([[0,0],[486,638],[98,735],[0,735]]),
                        np.array([[720,1280],[720,802],[0,802],[0,1280]])]
        add_poly = [np.array([[0,735],[98,735],[687,802],[0,802]])]               
        myimg = cv2.fillPoly(img,external_poly, (0,0,0))
        frame = cv2.fillPoly( myimg,add_poly, (0,0,0))





        if return_value:
            # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)  #rotate the video for mobile videos
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_num += 1
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break
    
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        if FLAGS.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=100,
            max_total_size=100,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
       )

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)
        
        pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]
        # print(pred_bbox[2])
        out_boxes, out_scores, out_classes, num_boxes = pred_bbox

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        # allowed_classes = list(class_names.values())
        
        # custom allowed classes (uncomment line below to allow detections for only SELECTED DETECTION CLASSES)
        allowed_classes = ['person','car','truck','bus','motorbike']
        # allowed_classes = ['car']



# UNCOMMENT THIS TO CALCULATE THE SPEED
        # #################################################################################################################################
        # #Calculting the distance   xmin, ymin, xmax, ymax
        # cv2.line(img=frame, pt1=(595,940),pt2=(1567,940),  color=(0, 0, 180), thickness=3, lineType=8, shift=0)
        # for i, b in zip(out_boxes,out_classes):
        #     f = 1460 ## Focal length of the camera
        #     if i[0]>1 and i[1]>105 and i[2]<1920 and i[3]<910 and int(b) == 2:
        #         a = int(i[0])
        #         c = int(i[2])
        #         if 1018 and 1053 in range(a,c):
        #     # if i[0]>1 and i[1]>105 and i[2]<1920 and i[3]<940 and int(b) == 2:    
        #             wpix = i[2] - i[0]
        #             w = 1.7  #car width
        #             # D = round((f*w)/wpix, 2)
        #             # h = 1.2        #ORIGINAL === 1.6            # Most vehicles have a size that ranges from 1.5 – 1.8 meters high and widths of 1.6-1.7 meters.
                    
        #             d_original =round((f*w)/wpix,2)
        #             d = d_original - 3    #2.3 from the camera to the front of the car 2.2 is for yolo car 
        #             d = round(d,2)
        #             print("{} meters".format(d_original),end = ",")
        #             print("{} meters".format(d))
        #             cv2.putText(frame, "{}m".format(d), (int(i[0]+ ((int(i[2]-int(i[0]))/2))), int(i[1])-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36,255,12), 2)


        #     # elif i[0]>1 and i[1]>105 and i[2]<1920 and i[3]<940 and int(b) == 3:
        #     #     # wpix = i[3] - i[1]
        #     #     # w = 1.6  #car width
        #     #     # D = round((f*w)/wpix, 2)
        #     #     h = 0.75           # Most motorbikes height 75 cm 
        #     #     d_original =round((f*h)/wpix,2)
        #     #     d = d_original - 2.3 -1.1  # 2.3 from the camera to the front of the car 1.1 is for yolo motorbike 
        #     #     d = round(d,2)
        #     #     print("{} meters".format(d_original),end = ",")
        #     #     print("{} meters".format(d))
        #     #     cv2.putText(frame, "{}m".format(d), (int(i[0]+ ((int(i[2]-int(i[0]))/2))), int(i[1])-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36,255,12), 2)

        #     # # ##Because bus is too long I will measure with width 
        #     # elif i[0]>1 and i[1]>105 and i[2]<1920 and i[3]<940 and int(b) == 5:
        #     #     wpix = i[2] - i[0]
        #     #     # w = 1.6  #
        #     #     # D = round((f*w)/wpix, 2)
        #     #     # h = 3.5           # Most buses height 4.3 meters
        #     #     w = 2.3
        #     #     d_original =round((f*w)/wpix,2)
        #     #     d = d_original - 2.3 # 2.3 from the camera to the front of the car  BUSSS
        #     #     d = round(d,2) 
        #     #     print("{} meters".format(d_original),end = ",")
        #     #     print("{} meters".format(d))
        #     #     cv2.putText(frame, "{}m".format(d), (int(i[0]+ ((int(i[2]-int(i[0]))/2))), int(i[1])-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36,255,12), 2)
              
        #     # elif i[0]>1 and i[1]>105 and i[2]<1920 and i[3]<940 and int(b) == 6:
        #     #     wpix = i[3] - i[1]
        #     #     # w = 1.6  #car width
        #     #     # D = round((f*w)/wpix, 2)
        #     #     h = 4.3          # Most train height 4.3 meters 
        #     #     d_original =round((f*h)/wpix,2)
        #     #     d = d_original - 2.3  # 2.3 from the camera to the front of the car, NO TRAINNNNNNNNN
        #     #     d = round(d,2)
        #     #     print("{} meters".format(d_original),end = ",")
        #     #     print("{} meters".format(d))
        #     #     cv2.putText(frame, "{}m".format(d), (int(i[0]+ ((int(i[2]-int(i[0]))/2))), int(i[1])-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36,255,12), 2)

        #     # #Because truck is too long I will measure with width     
        #     elif i[0]>1 and i[1]>105 and i[2]<1920 and i[3]<910 and int(b) == 7:
        #         a = int(i[0])
        #         c = int(i[2])
        #         if 1018 and 1053 in range(a,c):
        #             wpix = i[2] - i[0]
        #             # w = 2.7  #Most truck height 2.7
        #             # D = round((f*w)/wpix, 2)
        #             # h = 2.5        # Most truck height 4.3 meters
        #             w = 1.7
        #             d_original =round((f*w)/wpix,2)
        #             d = d_original - 3  # 2.3 from the camera to the front of the car 5 is for yolo TRUCKCKKKKKKK
        #             d = round(d,2)
        #             print("{} meters".format(d_original),end = ",")
        #             print("{} meters".format(d))
        #             cv2.putText(frame, "{}m".format(d), (int(i[0]+ ((int(i[2]-int(i[0]))/2))), int(i[1])-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36,255,12), 2)
            
        #     else:
        #         None
        # ###################################################################################################################################



        # if crop flag is enabled, crop each detection and save it as new image
        if FLAGS.crop:
            crop_rate = 150 # capture images every so many frames (ex. crop photos every 150 frames)
            crop_path = os.path.join(os.getcwd(), 'detections', 'crop', video_name)
            try:
                os.mkdir(crop_path)
            except FileExistsError:
                pass
            if frame_num % crop_rate == 0:
                final_path = os.path.join(crop_path, 'frame_' + str(frame_num))
                try:
                    os.mkdir(final_path)
                except FileExistsError:
                    pass          
                crop_objects(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), pred_bbox, final_path, allowed_classes)
            else:
                pass

        if FLAGS.count:
            # count objects found
            counted_classes = count_objects(pred_bbox, by_class = True, allowed_classes=allowed_classes)
            # loop through dict and print
            for key, value in counted_classes.items():
                print("Number of {}s: {}".format(key, value))
            image = utils.draw_bbox(frame, pred_bbox, FLAGS.info, counted_classes, allowed_classes=allowed_classes, read_plate=FLAGS.plate)
        else:
            image = utils.draw_bbox(frame, pred_bbox, FLAGS.info, allowed_classes=allowed_classes, read_plate=FLAGS.plate)


        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
        result = np.asarray(image)
        cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
        result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if not FLAGS.dont_show:
            cv2.imshow("result", result)
        
        if FLAGS.output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
