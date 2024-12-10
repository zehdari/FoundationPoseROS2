import sys
sys.path.append('./FoundationPose')
sys.path.append('./FoundationPose/nvdiffrast')

import rclpy
from rclpy.node import Node
from estimater import *
import cv2
import numpy as np
import trimesh
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Bool
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation as R
from ultralytics import YOLO
from rcl_interfaces.msg import SetParametersResult
import os

COLORS = [(0, 255, 0)]

# Set up Foundation pose
original_init = FoundationPose.__init__
original_register = FoundationPose.register

def modified_init(self, model_pts, model_normals, symmetry_tfs=None, mesh=None, scorer=None, refiner=None, glctx=None, debug=0, debug_dir='./FoundationPose'):
    original_init(self, model_pts, model_normals, symmetry_tfs, mesh, scorer, refiner, glctx, debug, debug_dir)
    self.is_register = False

def modified_register(self, K, rgb, depth, ob_mask, iteration):
    pose = original_register(self, K, rgb, depth, ob_mask, iteration)
    self.is_register = True
    return pose

FoundationPose.__init__ = modified_init
FoundationPose.register = modified_register

# Node for pose est
class PoseEstimationNode(Node):
    def __init__(self):
        super().__init__('pose_estimation_node')

        # Define object configurations, gunna be in a config later
        self.object_configs = {
            'bowl': {
                'mesh_path': '~/FoundationPoseROS2/demo_data/bowl/bowl_translated.obj',
                'yolo_class': 'bowl'
            },
            'mug': {
                'mesh_path': '~/FoundationPoseROS2/demo_data/mug/mug16.obj',
                'yolo_class': 'cup'
            },
            'scissors': {
                'mesh_path': '~/FoundationPoseROS2/demo_data/scissors/scissors16.obj',
                'yolo_class': 'scissors'
            }
        }

        # Parameters
        self.declare_parameter('enable_visualization', False)
        self.declare_parameter('debug', False)
        self.declare_parameter('target_object', 'bowl')  # Default to bowl
        
        self.enable_visualization = self.get_parameter('enable_visualization').value
        self.target_object = self.get_parameter('target_object').value
        self.debug = self.get_parameter('debug').value

        if self.target_object not in self.object_configs:
            self.get_logger().error(f"Invalid target object: {self.target_object}. Using default: bowl")
            self.target_object = 'bowl'
        
        self.debug_dir = os.path.join(os.getcwd(), 'foundation_pose_debug')
        os.makedirs(self.debug_dir, exist_ok=True)
        
        # Load object-specific configuration
        self.load_object_config()
        
        # Subscribers
        self.image_sub = self.create_subscription(Image, '/zed/zed_node/left/image_rect_color', self.image_callback, 10)
        self.depth_sub = self.create_subscription(Image, '/zed/zed_node/depth/depth_registered', self.depth_callback, 10)
        self.info_sub = self.create_subscription(CameraInfo, '/zed/zed_node/left/camera_info', self.camera_info_callback, 10)
        
        # Publishers
        self.pose_pub = self.create_publisher(PoseStamped, '/object_pose', 10)
        self.vis_pub = self.create_publisher(Image, '/pose_visualization', 10) 
        self.marker_pub = self.create_publisher(MarkerArray, '/object_bbox', 10)
        self.tracking_pub = self.create_publisher(Bool, '/object_tracking_active', 10)
        
        # Parameter change callback
        self.add_on_set_parameters_callback(self.parameter_callback)
        
        self.bridge = CvBridge()
        self.depth_image = None
        self.color_image = None
        self.cam_K = None
        self.current_score = None
        self.current_stamp = None
        self.tracking_active = False
        self.prev_tracking_state = None
        
        self.scorer = ScorePredictor()
        self.refiner = PoseRefinePredictor()
        self.glctx = dr.RasterizeCudaContext()
        
        self.det_model = YOLO("yolo11n.pt")
        
        self.object_mask = None
        self.initialized = False
        self.mask_color = COLORS[0]
        
        self.frame_count = 0
        self.validation_interval = 10
        self.iou_threshold = 0.3

        # Add these new instance variables for tracking bboxes
        self.last_det_bbox = None
        self.last_track_bbox = None
        self.bbox_timestamp = None
        self.bbox_timeout = 3.0
        
        # Initialize pose estimator
        self.initialize_pose_estimator()

    # Load config for the object
    def load_object_config(self):
        config = self.object_configs[self.target_object]
        self.mesh_path = os.path.expanduser(config['mesh_path'])
        self.target_class_name = config['yolo_class']
        self.mesh = trimesh.load(self.mesh_path)
        self.bound, extents = trimesh.bounds.oriented_bounds(self.mesh)
        self.bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2, 3)

    # Initialize or reinitialize the pose estimator with current object configuration
    def initialize_pose_estimator(self):
        self.pose_est = FoundationPose(
            model_pts=self.mesh.vertices,
            model_normals=self.mesh.vertex_normals,
            mesh=self.mesh,
            scorer=self.scorer,
            refiner=self.refiner,
            glctx=self.glctx,
            debug_dir=self.debug_dir
        )
        self.pose_est.is_register = False
        self.current_score = None
        self.object_mask = None
        self.tracking_active = False

    # Param updates
    def parameter_callback(self, params):
        success = True
        for param in params:
            if param.name == 'target_object' and param.value != self.target_object:
                if param.value in self.object_configs:
                    try:
                        self.target_object = param.value
                        self.get_logger().info(f"Switching to target object: {self.target_object}")
                        self.load_object_config()
                        self.initialize_pose_estimator()
                    except Exception as e:
                        self.get_logger().error(f"Failed to switch object: {str(e)}")
                        success = False
                else:
                    self.get_logger().error(f"Invalid target object: {param.value}")
                    success = False
            elif param.name == 'enable_visualization' and param.value != self.enable_visualization:
                try:
                    self.enable_visualization = param.value
                    self.get_logger().info(f"Setting Visualization to : {self.enable_visualization}")
                except Exception as e:
                    self.get_logger().error(f"Failed to set visualization: {str(e)}")
                    success = False
            elif param.name == 'debug' and param.value != self.debug:
                try:
                    self.debug = param.value
                    self.get_logger().info(f"Setting Debug to : {self.enable_visualization}")
                except Exception as e:
                    self.get_logger().error(f"Failed to set debug: {str(e)}")
                    success = False

        return SetParametersResult(successful=success)

    # Cam info
    def camera_info_callback(self, msg):
        if self.cam_K is None:
            self.cam_K = np.array(msg.k).reshape((3, 3))

    # Img sub callback
    def image_callback(self, msg):
        try:
            self.current_stamp = msg.header.stamp
            bgr_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.color_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            self.get_logger().error(f'Error converting image message: {str(e)}')

    # Depth sub callback
    def depth_callback(self, msg):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, "32FC1")
            self.process_images()
        except Exception as e:
            self.get_logger().error(f'Error converting depth message: {str(e)}')

    # Publish the current tracking state if it has changed, gunna be used for autonomy later
    def publish_tracking_state(self):
        if self.prev_tracking_state is None or self.tracking_active != self.prev_tracking_state:
            tracking_msg = Bool()
            tracking_msg.data = self.tracking_active
            self.tracking_pub.publish(tracking_msg)
            
            state_str = "started" if self.tracking_active else "lost"
            self.get_logger().info(f"Tracking {state_str}")
            
            self.prev_tracking_state = self.tracking_active

    # Get iou between FP 3d mask and yolo bbox mask
    def calculate_iou(self, mask1, mask2):
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return intersection / union if union > 0 else 0
        
    # Create a binary mask from projected mesh vertices, was here from when i was using seg instead of bbox
    def get_projected_mask(self, pose, image_shape):
        if not self.debug:
            return np.zeros(image_shape[:2], dtype=np.uint8)
            
        self.get_logger().info("Starting projected mask calc")
        
        # Create coordinate transform matrix to swap X and Z, and flip Z, needed this for it to work, prob going to fix in models later
        coord_transform = np.array([
            [0, 0, -1, 0],  # X = -old Z (flip Z)
            [0, 1, 0, 0],   # Y stays the same
            [1, 0, 0, 0],   # Z = old X
            [0, 0, 0, 1]
        ])
        
        # Initialize mask
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        
        # Create homogeneous coordinates
        verts = np.hstack((self.mesh.vertices, np.ones((len(self.mesh.vertices), 1))))
        transformed_verts = (pose @ coord_transform @ verts.T).T[:, :3]
        
        # Project all points at once
        points_2d = (self.cam_K @ transformed_verts.T).T
        points_2d = points_2d[:, :2] / points_2d[:, 2:]
        points_2d = points_2d.astype(np.int32)
        
        # Get all face points and fill polygons in one go
        face_points = points_2d[self.mesh.faces]
        cv2.fillPoly(mask, face_points, 1)
        
        self.get_logger().info("Finishing projected mask calc")
        return mask > 0

    # Project 3D object vertices to 2D and get the enclosing bounding box.
    # Uses the same coordinate transform as mask projection for consistency.
    # Returns bbox in [x1, y1, x2, y2] format.
    def get_projected_bbox(self, pose, image_shape):

        # Create coordinate transform matrix to swap X and Z, and flip Z, again gunna hopefully fix this in 3d model later
        coord_transform = np.array([
            [0, 0, -1, 0],  # X = -old Z (flip Z)
            [0, 1, 0, 0],   # Y stays the same
            [1, 0, 0, 0],   # Z = old X
            [0, 0, 0, 1]
        ])
        
        # Create homogeneous coordinates
        verts = np.hstack((self.mesh.vertices, np.ones((len(self.mesh.vertices), 1))))
        transformed_verts = (pose @ coord_transform @ verts.T).T[:, :3]
        
        # Project all points at once
        points_2d = (self.cam_K @ transformed_verts.T).T
        uv = points_2d[:, :2] / points_2d[:, 2:3]
        
        # Get bounding rectangle of projected points
        x_min = max(0, int(np.min(uv[:, 0])))
        y_min = max(0, int(np.min(uv[:, 1])))
        x_max = min(image_shape[1], int(np.max(uv[:, 0])))
        y_max = min(image_shape[0], int(np.max(uv[:, 1])))
        
        return np.array([x_min, y_min, x_max, y_max])


    # Calculate IoU between two bounding boxes in [x1, y1, x2, y2] format
    def calculate_bbox_iou(self, bbox1, bbox2):
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0

    # Create a binary mask from bounding box coordinates
    def create_mask_from_bbox(self, bbox, image_shape):
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        x1, y1, x2, y2 = map(int, bbox)
        mask[y1:y2, x1:x2] = 1
        return mask

    # Where the magic happens, process the image and feed it to FP, if iou is bad, restart the slow FP node with yolo
    def process_images(self):
        if self.color_image is None or self.depth_image is None or self.cam_K is None:
            return

        H, W = self.color_image.shape[:2]
        color = cv2.resize(self.color_image, (W, H), interpolation=cv2.INTER_NEAREST)
        depth = cv2.resize(self.depth_image, (W, H), interpolation=cv2.INTER_NEAREST)
        
        depth[(depth < 0.3) | (depth > 20.0)] = 0 # Zed depth range

        run_detection = False
        if self.pose_est.is_register: # ie if FP has started tracking model
            extra = {}
            # This runs the tracking FP model
            pose = self.pose_est.track_one(rgb=color, depth=depth, K=self.cam_K, iteration=2, extra=extra)
            self.tracking_active = True
            
            if pose is None: # Shouldnt happen, but here just in case
                self.get_logger().warn("Lost tracking - reverting to detection mode")
                self.pose_est.is_register = False
                self.current_score = None
                run_detection = True
                self.tracking_active = False

            # Validating every n frames checking iou to see if we lost tracking using yolo vs 3d model, will restart slow FP model if so
            elif self.frame_count % self.validation_interval == 0:
                results = self.det_model.predict(color, conf=0.5, iou=0.7)
                if results and len(results[0]) > 0:
                    for r in results[0]:
                        cls = int(r.boxes.cls.item())
                        class_name = self.det_model.names[cls]
                        if class_name.lower() == self.target_class_name.lower():
                            # Get YOLO detection bbox
                            det_bbox = r.boxes.xyxy[0].cpu().numpy()
                            self.last_det_bbox = det_bbox
                            
                            # Get tracked object's projected bbox using full mesh vertices
                            center_pose = pose @ np.linalg.inv(self.bound)
                            track_bbox = self.get_projected_bbox(center_pose, color.shape)
                            self.last_track_bbox = track_bbox
                            
                            # Update timestamp
                            self.bbox_timestamp = self.get_clock().now()
                            
                            # Calculate IoU between bounding boxes
                            iou = self.calculate_bbox_iou(det_bbox, track_bbox)
                            
                            # Debug logging
                            self.get_logger().info(f"Detection bbox: {det_bbox}")
                            self.get_logger().info(f"Projected bbox: {track_bbox}")
                            self.get_logger().info(f"IoU: {iou:.3f}")
                            
                            if iou < self.iou_threshold:
                                self.get_logger().warn(f"Low IoU ({iou:.2f}) - reverting to detection")
                                self.pose_est.is_register = False
                                self.current_score = None
                                run_detection = True
                                self.tracking_active = False
                            break
        else:
            run_detection = True
            self.tracking_active = False

        self.publish_tracking_state()
        # If this runs, it means we're running first node in FP, the slow but accurate one, needs yolo unlike tracking node
        if run_detection:
            results = self.det_model.predict(color, conf=0.5, iou=0.7)
            object_detected = False
            self.tracking_active = False
            
            # Go through yolo detections, make mask from bbox since FP needs a mask
            if results and len(results[0]) > 0:
                for r in results[0]:
                    cls = int(r.boxes.cls.item())
                    class_name = self.det_model.names[cls]
                    
                    if class_name.lower() == self.target_class_name.lower():
                        bbox = r.boxes.xyxy[0].cpu().numpy()
                        self.object_mask = self.create_mask_from_bbox(bbox, (H, W))
                        object_detected = True
                        break
            
            if not object_detected:
                self.get_logger().info(f"No {self.target_class_name} detected")
                return

            # This is what FP main node uses to run
            pose = self.pose_est.register(K=self.cam_K, rgb=color, depth=depth, 
                                        ob_mask=self.object_mask, iteration=4)
            if pose is not None:
                self.current_score = float(self.pose_est.scores[0])
            else:
                self.current_score = None

        if pose is not None:
            center_pose = pose @ np.linalg.inv(self.bound)
            
            # Always publish pose and markers
            self.publish_pose(center_pose)
            
            # Only process and publish visualization if enabled
            if self.enable_visualization and self.vis_pub is not None:
                vis_img = color.copy()
                
                if not self.pose_est.is_register and self.object_mask is not None:
                    mask_overlay = np.zeros_like(vis_img)
                    mask_overlay[self.object_mask > 0] = self.mask_color
                    cv2.addWeighted(vis_img, 1, mask_overlay, 0.3, 0, vis_img)
                
                vis_img = self.visualize_pose(vis_img, center_pose)
                
                position = center_pose[:3, 3]
                rotation = R.from_matrix(center_pose[:3, :3]).as_euler('xyz', degrees=True)
                
                info_text = [
                    f"Mode: {'Tracking' if self.pose_est.is_register else 'Detection'}",
                    f"Position (x,y,z): {position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}",
                    f"Rotation (deg): {rotation[0]:.1f}, {rotation[1]:.1f}, {rotation[2]:.1f}"
                ]
                
                for i, text in enumerate(info_text):
                    cv2.putText(vis_img, text, (10, 30 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 
                               1, (0, 255, 0), 2, cv2.LINE_AA)
                
                vis_img_bgr = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
                
                try:
                    vis_msg = self.bridge.cv2_to_imgmsg(vis_img_bgr, encoding="bgr8")
                    vis_msg.header.stamp = self.current_stamp
                    vis_msg.header.frame_id = "zed_left_camera_frame"
                    self.vis_pub.publish(vis_msg)
                except Exception as e:
                    self.get_logger().error(f'Error publishing visualization: {str(e)}')

        self.frame_count += 1 # Used for iou validation every n frames

    # BBox marker for rviz
    def create_bbox_marker(self, pose_array, marker_id):
        marker = Marker()
        marker.header.frame_id = "zed_left_camera_frame"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "object_bbox"
        marker.id = marker_id
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.005  
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        
        # Convert bbox corners to points in the world frame
        corners = []
        for x in [self.bbox[0][0], self.bbox[1][0]]:
            for y in [self.bbox[0][1], self.bbox[1][1]]:
                for z in [self.bbox[0][2], self.bbox[1][2]]:
                    corners.append([x, y, z, 1])
        
        corners = np.array(corners)
        
        # Create transformation matrix from pose
        transform = np.eye(4)
        transform[:3, 3] = pose_array[:3]  # Pos
        transform[:3, :3] = R.from_quat(pose_array[3:7]).as_matrix()  # Orientation
                
        # Transform corners
        transformed_corners = (transform @ corners.T).T[:, :3]
        
        # Define lines connecting corners
        lines = [
            (0, 1), (1, 3), (3, 2), (2, 0),  # Bottom face
            (4, 5), (5, 7), (7, 6), (6, 4),  # Top face
            (0, 4), (1, 5), (2, 6), (3, 7)   # Connecting lines
        ]
        
        # Add points for each line
        for start_idx, end_idx in lines:
            start = transformed_corners[start_idx]
            end = transformed_corners[end_idx]
            
            p1 = Point()
            p1.x, p1.y, p1.z = start
            p2 = Point()
            p2.x, p2.y, p2.z = end
            
            marker.points.extend([p1, p2])
        
        return marker
    
    # Publishes the pos, as well as the rviz markers
    def publish_pose(self, center_pose):
        if self.current_stamp is None:
            self.get_logger().warn("No timestamp available, skipping pose publication")
            return

        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.current_stamp
        pose_msg.header.frame_id = "zed_left_camera_frame"

        cv_to_ros = np.array([
            [0, 0, 1, 0],
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1]
        ])

        transformed_pose = cv_to_ros @ center_pose

        position = transformed_pose[:3, 3]
        rotation = R.from_matrix(transformed_pose[:3, :3])
        quaternion = rotation.as_quat()

        pose_msg.pose.position.x = position[0]
        pose_msg.pose.position.y = position[1]
        pose_msg.pose.position.z = position[2]
        pose_msg.pose.orientation.x = quaternion[0]
        pose_msg.pose.orientation.y = quaternion[1]
        pose_msg.pose.orientation.z = quaternion[2]
        pose_msg.pose.orientation.w = quaternion[3]
        
        self.pose_pub.publish(pose_msg)

        marker_array = MarkerArray()
        pose_array = np.concatenate((position, quaternion))
        bbox_marker = self.create_bbox_marker(pose_array, 0)
        bbox_marker.header.stamp = self.current_stamp
        marker_array.markers.append(bbox_marker)
        self.marker_pub.publish(marker_array)

    
    def visualize_pose(self, image, center_pose):
        """Add visualization with corrected coordinate system."""
        vis = image.copy()
        
        # Draw 3D bounding box and axes regardless of debug mode
        vis = draw_posed_3d_box(self.cam_K, img=vis, ob_in_cam=center_pose, bbox=self.bbox)
        vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.1, K=self.cam_K, thickness=3, transparency=0, is_input_rgb=True)
        
        # Debug shows the 3d mask used for calculating the bbox used in iou from the model, visually see the model in 2d essentially
        if self.debug:
            # Create projected mask
            projected_mask = self.get_projected_mask(center_pose, image.shape)
            
            # Create mask overlay (semi-transparent blue)
            mask_overlay = np.zeros_like(image)
            mask_overlay[projected_mask] = [0, 0, 255] 
            
            # Blend mask with original image
            vis = cv2.addWeighted(vis, 1, mask_overlay, 0.3, 0)
        
        # The following visualization code runs regardless of debug mode
        # Check if bboxes are still valid (not too old)
        current_time = self.get_clock().now()
        boxes_valid = (self.bbox_timestamp is not None and 
                    (current_time - self.bbox_timestamp).nanoseconds / 1e9 < self.bbox_timeout)
        
        if boxes_valid:
            # Draw detection bbox in red if available
            if self.last_det_bbox is not None:
                cv2.rectangle(vis, 
                            (int(self.last_det_bbox[0]), int(self.last_det_bbox[1])),
                            (int(self.last_det_bbox[2]), int(self.last_det_bbox[3])),
                            (255, 0, 0), 2)
                # Add label
                cv2.putText(vis, 'Detection', 
                        (int(self.last_det_bbox[0]), int(self.last_det_bbox[1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # Draw tracking bbox in green if available
            if self.last_track_bbox is not None:
                cv2.rectangle(vis,
                            (int(self.last_track_bbox[0]), int(self.last_track_bbox[1])),
                            (int(self.last_track_bbox[2]), int(self.last_track_bbox[3])),
                            (0, 255, 0), 2)
                # Add label
                cv2.putText(vis, 'Tracking', 
                        (int(self.last_track_bbox[0]), int(self.last_track_bbox[1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Draw IoU value if both boxes are present
                if self.last_det_bbox is not None:
                    iou = self.calculate_bbox_iou(self.last_det_bbox, self.last_track_bbox)
                    cv2.putText(vis, f'IoU: {iou:.2f}', 
                            (10, vis.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return vis
    
    # Ros things
def main(args=None):
    rclpy.init(args=args)
    node = PoseEstimationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

# The classic
if __name__ == '__main__':
    main()
