from styx_msgs.msg import TrafficLight
import rospy
import tensorflow as tf
import numpy as np
from PIL import Image

class TLClassifier(object):
    def __init__(self):
	graph_obj = tf.Graph()
	with graph_obj.as_default():
      		od_graph_def = tf.GraphDef()
	od_graph_def.ParseFromString(tf.gfile.GFile('model_sim.pb', 'rb').read())
	tf.import_graph_def(od_graph_def, name='')

	self.graph                     = graph_obj
	self.input_tensor              = graph_obj.get_tensor_by_name('image_tensor:0')
	self.bounding_boxes_tensor     = graph_obj.get_tensor_by_name('detection_boxes:0')
	self.predicted_score_tensor    = graph_obj.get_tensor_by_name('detection_scores:0')
	self.predicted_classes_tensor  = graph_obj.get_tensor_by_name('detection_classes:0')
	self.predicted_obj_num_tensor  = graph_obj.get_tensor_by_name('num_detections:0')


    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
	(im_width, im_height) = image.size
	img_data = np.expand_dims(image.reshape((im_height, im_width, 3)).astype(np.uint8))

	with self.graph.as_default():
        	with tf.Session(graph=self.graph) as sess :
			(boxes, scores, classes, num) = sess.run(
                  	[self.bounding_boxes_tensor, self.predicted_score_tensor, self.predicted_classes_tensor, self.predicted_obj_num_tensor],
                  					{self.input_tensor: img_data})
			rospy.logdebug(classes)

			

        return TrafficLight.UNKNOWN
