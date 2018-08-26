from styx_msgs.msg import TrafficLight
import rospy
import tensorflow as tf
import numpy as np
import cv2

class TLClassifier(object):
    def __init__(self):
		self.graph = tf.Graph()
		with tf.Session(graph=self.graph) as sess:
			od_graph_def = tf.GraphDef()
			od_graph_def.ParseFromString(tf.gfile.GFile('light_classification/model_sim.pb', 'rb').read())
			tf.import_graph_def(od_graph_def, name='')

		self.input_tensor              = self.graph.get_tensor_by_name('image_tensor:0')
		#self.bounding_boxes_tensor     = self.graph.get_tensor_by_name('detection_boxes:0')
		self.predicted_score_tensor    = self.graph.get_tensor_by_name('detection_scores:0')
		self.predicted_classes_tensor  = self.graph.get_tensor_by_name('detection_classes:0')
		self.predicted_obj_num_tensor  = self.graph.get_tensor_by_name('num_detections:0')


    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        image = np.array(image)

        with self.graph.as_default():
            with tf.Session(graph=self.graph) as sess :
                ( scores, classes, num) = sess.run(
                    [self.predicted_score_tensor, self.predicted_classes_tensor, self.predicted_obj_num_tensor],
                            {self.input_tensor: [image]})
                #rospy.logwarn(classes)
                #rospy.logwarn(scores)

			

        return TrafficLight.UNKNOWN
