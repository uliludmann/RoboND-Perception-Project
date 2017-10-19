#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml


# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

# Exercise-2 TODOs:
    input_point_cloud_pub.publish(pcl_msg)
    # TODO: Convert ROS msg to PCL data

    cloud = ros_to_pcl(pcl_msg)
    
    # TODO: Statistical Outlier Filtering
    outlier_filter = cloud.make_statistical_outlier_filter()
    outlier_filter.set_mean_k(10) #set number of neighboring points to analyze
    outlier_filter.set_std_dev_mul_thresh(0.003) #0.05 any point with a mean distance larger rhan global(mean distance+x * std dev) will be considered as a outlier
    #call the filter
    cloud_filtered = outlier_filter.filter()
    statistical_filter_pub.publish(pcl_to_ros(cloud_filtered))
    # TODO: Voxel Grid Downsampling
    vox = cloud_filtered.make_voxel_grid_filter()
    LEAF_SIZE = 0.01
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    vox_filtered = vox.filter()
    vox_filter_pub.publish(pcl_to_ros(vox_filtered))
    
    # TODO: PassThrough Filter
    # z axis
    z_passthrough = vox_filtered.make_passthrough_filter()
    filter_axis = 'z'
    z_passthrough.set_filter_field_name(filter_axis)
    axis_min = 0.55
    axis_max = 0.8
    z_passthrough.set_filter_limits(axis_min, axis_max)
    passthrough_filtered = z_passthrough.filter()

    # x axis -> points away from fobot

    x_passthrough = passthrough_filtered.make_passthrough_filter()
    filter_axis = 'x'
    x_passthrough.set_filter_field_name(filter_axis)
    x_passthrough.set_filter_limits(0.4, 0.7)
    passthrough_filtered = x_passthrough.filter()

    # y axis
    """
    y_passthrough = passthrough_filtered.make_passthrough_filter()
    y_passthrough.set_filter_field_name('y')
    y_passthrough.set_filter_limits(0.1, 30)
    passthrough_filtered = y_passthrough.filter()
    """
    
    passthrough_filter_pub.publish(pcl_to_ros(passthrough_filtered))

    # TODO: RANSAC Plane Segmentation

    seg = passthrough_filtered.make_segmenter()
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    max_distance = 0.01
    seg.set_distance_threshold(max_distance)

    # TODO: Extract inliers and outliers
    inliers, coeficcients = seg.segment()
    extracted_inliers = passthrough_filtered.extract(inliers, negative = False)
    extracted_outliers = passthrough_filtered.extract(inliers, negative = True)
    cloud_table = extracted_inliers
    cloud_objects = extracted_outliers

    ransac_filter_pub.publish(pcl_to_ros(cloud_table))
    ransac_objects_pub.publish(pcl_to_ros(cloud_objects))

    # TODO: Euclidean Clustering
    white_cloud = XYZRGB_to_XYZ(cloud_objects)
    tree = white_cloud.make_kdtree()
    ec = white_cloud.make_EuclideanClusterExtraction()
    ec.set_ClusterTolerance(0.05)
    ec.set_MinClusterSize(50)
    ec.set_MaxClusterSize(700)
    ec.set_SearchMethod(tree)
    cluster_indices = ec.Extract()

    print("found clusters: ", np.array(cluster_indices).shape)

    # TODO: Create Cluster-Mask Point Cloud to visualize each cluster separately
    cluster_color = get_color_list(len(cluster_indices))
    color_cluster_point_list = []
    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0],
                white_cloud[indice][1],
                white_cloud[indice][2],
                rgb_to_float(cluster_color[j])]
                )
    segmented_cloud = pcl.PointCloud_PointXYZRGB()
    segmented_cloud.from_list(color_cluster_point_list)
    
    # TODO: Convert PCL data to ROS messages
    # TODO: Publish ROS messages
    
    segmented_cloud_pub.publish(pcl_to_ros(segmented_cloud))

# Exercise-3 TODOs:
    detected_objects_labels = []
    detected_objects = []

    # Classify the clusters! (loop through each detected cluster one at a time)
    for index, pts_list in enumerate(cluster_indices):

        
        # Grab the points for the cluster
        pcl_cluster = cloud_objects.extract(pts_list)
        cluster_publisher.publish(pcl_to_ros(pcl_cluster))

        ros_cluster = pcl_to_ros(pcl_cluster)
        # Compute the Assigociated feature vector

        chists = compute_color_histograms(ros_cluster, using_hsv = True)
        normals = get_normals(ros_cluster)
        nhists = compute_normal_histograms(normals)
        feature = np.concatenate((chists, nhists))

        # Make the prediction
        prediction = clf.predict(scaler.transform(feature.reshape(1, -1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)

        # Publish a label into RViz
        label_pos = list(white_cloud[pts_list[0]])
        label_pos[2] += .25
        object_markers_pub.publish(make_label(label, label_pos, index))

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = pcl_cluster
        detected_objects.append(do)


    # Publish the list of detected objects

    rospy.loginfo('Detected {} objects {}'.format(len(detected_objects_labels), detected_objects_labels))

    # Suggested location for where to invoke your pr2_mover() function within pcl_callback()
    # Could add some logic to determine whether or not your object detections are robust
    # before calling pr2_mover()
    
    try:
        pr2_mover(detected_objects)
    except rospy.ROSInterruptException:
        pass
    

# function to load parameters and request PickPlace service
def pr2_mover(object_list):

    # TODO: Initialize variables
    pick_list = []
    pick_group = []
    labels = []
    centroids = []

    not_found_objects = []

    dict_list = []

    # TODO: Get/Read parameters
    object_list_param = rospy.get_param('/object_list')
    dropbox_parameters = rospy.get_param('/dropbox')
    world_name = 1 #rospy.get_param("~world_name")


    print('Object list parameters %s' % (object_list_param))
    print('Dropbox parameters %s' % (dropbox_parameters))


    # TODO: Parse parameters into individual variables
    for pl_object in object_list_param:
        pick_list.append(pl_object['name'])
        pick_group.append(pl_object['group'])

    print('Pick list %s' %(pick_list))
    print('Pick group %s' %(pick_group))

    for found_object in object_list:
        labels.append(found_object.label)

    print(labels)

    for pick_object in pick_list:
        print('pick object', pick_object)
        try:
            index = labels.index(pick_object)
        except ValueError:
            not_found_objects.append(pick_object)
            print('Object %s not found. Continuing with next Object in pick list' %(pick_object))
            continue
        # TODO: Get the PointCloud for a given object and obtain it's centroid
        points_arr = object_list[index].cloud.to_array()
        center = np.mean(points_arr, axis=0)[:3]
        center_scalar = []
        pick_pose = Pose()
        for i in center:
            center_scalar.append(np.asscalar(i))
        pick_pose.position.x, pick_pose.position.y, pick_pose.position.z = center_scalar
        print(pick_pose)
        PICK_POSE = pick_pose

        centroids.append(center_scalar)

        target_dropbox = pick_group[index]

        TEST_SCENE_NUM = Int32()
        TEST_SCENE_NUM.data = world_name
        OBJECT_NAME = String()
        OBJECT_NAME.data = pick_object


        # TODO: Create 'place_pose' for the object
        PLACE_POSE = Pose()
        # TODO: Assign the arm to be used for pick_place

        WHICH_ARM = String()
        if target_dropbox == "green": 
            WHICH_ARM.data = "right"
            PLACE_POSE.position.x, PLACE_POSE.position.y, PLACE_POSE.position.z = [0, -0.71, 0.7 ]
        else:
            WHICH_ARM.data = "left"
            PLACE_POSE.position.x, PLACE_POSE.position.y, PLACE_POSE.position.z = [0, +0.71, 0.7]


        # TODO: Create a list of dictionaries (made with make_yaml_dict()) for later output to yaml format

        for i in range(0, len(object_list_param)):
            yaml_dict = make_yaml_dict(TEST_SCENE_NUM, WHICH_ARM, OBJECT_NAME, PICK_POSE, PLACE_POSE)
            dict_list.append(yaml_dict)

        # Wait for 'pick_place_routine' service to come up
       
        rospy.wait_for_service('pick_place_routine')

        try:
            pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)

            # TODO: Insert your message variables to be sent as a service request
            resp = pick_place_routine(TEST_SCENE_NUM, OBJECT_NAME, WHICH_ARM, PICK_POSE, PLACE_POSE)

            print ("Response: ",resp.success)

        except rospy.ServiceException, e:
            print "Service call failed: %s"%e


        """

    # TODO: Output your request parameters into output yaml file
        place_pose = Pose()
    rospy.loginfo("These Objects were not found on the table: %s" %(pick_object))
    #print("centroids list %s" %(centroids))


    ###


    #rospy.loginfo('Pick list {}, means {}'.format(len(detected_objects_labels), detected_objects_labels))
    """

    # TODO: Rotate PR2 in place to capture side tables for the collision map



if __name__ == '__main__':

    # TODO: ROS node initialization
    rospy.init_node('object_recognition_project', anonymous = True)


    # TODO: Create Subscribers
    pcl_sub = rospy.Subscriber("/pr2/world/points", PointCloud2, pcl_callback, queue_size = 10)


    # TODO: Create Publishers
    input_point_cloud_pub = rospy.Publisher("/input_point_cloud", PointCloud2, queue_size = 1)
    statistical_filter_pub = rospy.Publisher("/statistical_filter_pub", PointCloud2, queue_size = 1)
    vox_filter_pub = rospy.Publisher("/vox_filter_pub", PointCloud2, queue_size = 1)
    passthrough_filter_pub = rospy.Publisher("/passthrough_filter_pub", PointCloud2, queue_size = 1)
    ransac_filter_pub = rospy.Publisher("/ransac_filter_pub", PointCloud2, queue_size = 1)
    ransac_objects_pub = rospy.Publisher("/ransac_objects_pub", PointCloud2, queue_size = 1)
    cluster_publisher =rospy.Publisher("/cluster_publisher", PointCloud2, queue_size = 1)

    object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=1)

    segmented_cloud_pub = rospy.Publisher("/segmented_cloud", PointCloud2, queue_size = 1)


    # TODO: Load Model From disk
    model_filename = 'model_64bins_100samples_sigmoid.sav'
    model = pickle.load(open(model_filename, 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']

    # Initialize color_list
    get_color_list.color_list = []

    # TODO: Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()
