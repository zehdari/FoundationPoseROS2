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
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation as R
from ultralytics import YOLO
from cam_2_base_transform import transformation
import os
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 0, 0),
    (0, 128, 0),
    (0, 0, 128),
    (128, 128, 0),
    (128, 0, 128),
    (0, 128, 128),
]

original_init = FoundationPose.__init__
original_register = FoundationPose.register

def modified_init(self, model_pts, model_normals, symmetry_tfs=None, mesh=None, scorer=None, refiner=None, glctx=None, debug=0, debug_dir='./FoundationPose'):
    original_init(self, model_pts, model_normals, symmetry_tfs, mesh, scorer, refiner, glctx, debug, debug_dir)
    self.is_register = False
    self.last_pose = None

def modified_register(self, K, rgb, depth, ob_mask, iteration):
    pose = original_register(self, K, rgb, depth, ob_mask, iteration)
    self.is_register = True
    return pose

FoundationPose.__init__ = modified_init
FoundationPose.register = modified_register

class PoseEstimationNode(Node):
    def __init__(self):
        super().__init__('pose_estimation_node')
        
        # Add TF2 buffer and listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.debug_dir = os.path.join(os.getcwd(), 'foundation_pose_debug')
        os.makedirs(self.debug_dir, exist_ok=True)
        
        self.image_sub = self.create_subscription(Image, '/zed/zed_node/left/image_rect_color', self.image_callback, 10)
        self.depth_sub = self.create_subscription(Image, '/zed/zed_node/depth/depth_registered', self.depth_callback, 10)
        self.info_sub = self.create_subscription(CameraInfo, '/zed/zed_node/left/camera_info', self.camera_info_callback, 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/object_pose', 10)
        
        self.bridge = CvBridge()
        self.depth_image = None
        self.color_image = None
        self.cam_K = None
        
        self.mesh_path = os.path.expanduser("~/FoundationPoseROS2/demo_data/bowl/bowl.obj")
        self.mesh = trimesh.load(self.mesh_path)
        
        self.bound, extents = trimesh.bounds.oriented_bounds(self.mesh)
        self.bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2, 3)
        
        self.scorer = ScorePredictor()
        self.refiner = PoseRefinePredictor()
        self.glctx = dr.RasterizeCudaContext()
        
        self.seg_model = YOLO("yolov8n-seg.pt")
        
        self.pose_est = None
        self.object_mask = None
        self.initialized = False
        self.selected_mask = None
        self.preview_mask = None
        self.preview_color = None
        self.masks_with_labels = None
        self.waiting_for_selection = False
        self.base_vis_image = None
        self.target_class = None
        
        self.last_camera_pose = None
        self.base_frame = "map"
        self.camera_frame = "zed2_left_camera_frame"
        
        cv2.namedWindow('Select Mask', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Tracking', cv2.WINDOW_NORMAL)
        
        self.class_names = self.seg_model.names

    def get_camera_pose(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.camera_frame,
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=1.0)
            )
            
            translation = [transform.transform.translation.x,
                         transform.transform.translation.y,
                         transform.transform.translation.z]
            rotation = [transform.transform.rotation.x,
                       transform.transform.rotation.y,
                       transform.transform.rotation.z,
                       transform.transform.rotation.w]
            
            rot_matrix = R.from_quat(rotation).as_matrix()
            camera_pose = np.eye(4)
            camera_pose[:3, :3] = rot_matrix
            camera_pose[:3, 3] = translation
            
            return camera_pose
            
        except Exception as ex:
            self.get_logger().warn(f"Could not get camera pose: {str(ex)}")
            return None

    def camera_info_callback(self, msg):
        if self.cam_K is None:
            self.cam_K = np.array(msg.k).reshape((3, 3))

    def image_callback(self, msg):
        try:
            bgr_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.color_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            self.get_logger().error(f'Error converting image message: {str(e)}')

    def depth_callback(self, msg):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, "32FC1")
            self.process_images()
        except Exception as e:
            self.get_logger().error(f'Error converting depth message: {str(e)}')

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE or event == cv2.EVENT_LBUTTONDOWN:
            if self.waiting_for_selection and self.masks_with_labels is not None:
                self.preview_mask = None
                self.preview_color = None
                
                masks_at_point = []
                for mask_info in self.masks_with_labels:
                    mask = mask_info['mask']
                    if mask[y, x] > 0:
                        area = np.sum(mask > 0)
                        masks_at_point.append((area, mask, mask_info['color'], mask_info['class_id'], mask_info['class_name']))
                
                if masks_at_point:
                    masks_at_point.sort(key=lambda x: x[0])
                    self.preview_mask = masks_at_point[0][1]
                    self.preview_color = masks_at_point[0][2]
                    self.preview_class_id = masks_at_point[0][3]
                    self.preview_class_name = masks_at_point[0][4]
                
                self.update_selection_display()

    def update_selection_display(self):
        if self.base_vis_image is None or self.preview_mask is None:
            return
            
        vis_image = self.base_vis_image.copy()
        if self.preview_mask is not None:
            highlight = np.zeros_like(vis_image)
            highlight[self.preview_mask > 0] = self.preview_color
            kernel = np.ones((5,5), np.uint8)
            dilated_mask = cv2.dilate(self.preview_mask, kernel, iterations=1)
            edge_mask = dilated_mask - self.preview_mask
            highlight[edge_mask > 0] = (255, 255, 255)
            
            cv2.addWeighted(vis_image, 0.7, highlight, 0.3, 0, vis_image)
            
            cv2.putText(vis_image, f"Selected: {self.preview_class_name} - Press ENTER to confirm", 
                       (10, vis_image.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
        cv2.imshow('Select Mask', vis_image[..., ::-1])

    def create_mask_visualization(self, image, results):
        H, W = image.shape[:2]
        vis_image = image.copy()
        masks_info = []
        
        for idx, r in enumerate(results[0]):
            try:
                cls = int(r.boxes.cls.item())
                conf = float(r.boxes.conf.item())
                class_name = self.class_names[cls]
                
                original_mask = r.masks.data[0].cpu().numpy()
                mask = cv2.resize(original_mask, (W, H), interpolation=cv2.INTER_NEAREST)
                mask = (mask > 0).astype(np.uint8)
                
                color = COLORS[idx % len(COLORS)]
                
                colored_mask = np.zeros_like(vis_image)
                colored_mask[mask > 0] = color
                
                cv2.addWeighted(vis_image, 1, colored_mask, 0.3, 0, vis_image)
                
                M = cv2.moments(mask)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    
                    label = f'{class_name} ({conf:.2f})'
                    cv2.putText(vis_image, label, (cx-2, cy-2), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 4)
                    cv2.putText(vis_image, label, (cx, cy), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
                    
                    masks_info.append({
                        'mask': mask,
                        'centroid': (cx, cy),
                        'color': color,
                        'class_id': cls,
                        'class_name': class_name,
                        'confidence': conf
                    })
                
            except Exception as e:
                self.get_logger().warn(f"Error processing mask {idx}: {str(e)}")
                continue
                
        return vis_image, masks_info

    def process_images(self):
        if self.color_image is None or self.depth_image is None or self.cam_K is None:
            return

        current_camera_pose = self.get_camera_pose()
        if current_camera_pose is None:
            return

        H, W = self.color_image.shape[:2]
        color = cv2.resize(self.color_image, (W, H), interpolation=cv2.INTER_NEAREST)
        depth = cv2.resize(self.depth_image, (W, H), interpolation=cv2.INTER_NEAREST)
        
        depth[(depth < 0.3) | (depth > 20.0)] = 0

        if not self.initialized:
            if not self.waiting_for_selection:
                self.get_logger().info("Detecting objects...")
                results = self.seg_model.predict(color, conf=0.5, iou=0.7)
                if not results:
                    self.get_logger().warn("No objects detected")
                    return
                
                self.base_vis_image, self.masks_with_labels = self.create_mask_visualization(color, results)
                
                if not self.masks_with_labels:
                    self.get_logger().warn("No valid masks found")
                    return
                
                cv2.imshow('Select Mask', self.base_vis_image[..., ::-1])
                cv2.setMouseCallback('Select Mask', self.mouse_callback)
                self.waiting_for_selection = True
                self.get_logger().info("Hover over a mask and press ENTER to select it for tracking")
                
            key = cv2.waitKey(1)
            if key == 13 and self.preview_mask is not None:
                self.selected_mask = self.preview_mask
                self.object_mask = self.selected_mask
                self.mask_color = self.preview_color
                self.target_class = self.preview_class_id
                self.target_class_name = self.preview_class_name
                self.pose_est = FoundationPose(
                    model_pts=self.mesh.vertices,
                    model_normals=self.mesh.vertex_normals,
                    mesh=self.mesh,
                    scorer=self.scorer,
                    refiner=self.refiner,
                    glctx=self.glctx,
                    debug_dir=self.debug_dir
                )
                self.initialized = True
                cv2.destroyWindow('Select Mask')
                self.get_logger().info(f"Tracking initialized for {self.target_class_name}")
                self.waiting_for_selection = False
                
        else:
            if self.last_camera_pose is not None and hasattr(self.pose_est, 'last_pose') and self.pose_est.last_pose is not None:
                camera_motion = np.linalg.inv(self.last_camera_pose) @ current_camera_pose
                initial_guess = camera_motion @ self.pose_est.last_pose
                self.pose_est.set_initial_pose(initial_guess)

            results = self.seg_model.predict(color, conf=0.5, iou=0.7)
            if results:
                for r in results[0]:
                    cls = int(r.boxes.cls.item())
                    if cls == self.target_class:
                        original_mask = r.masks.data[0].cpu().numpy()
                        mask = cv2.resize(original_mask, (W, H), interpolation=cv2.INTER_NEAREST)
                        self.object_mask = (mask > 0).astype(np.uint8)
                        break
            
            if self.pose_est.is_register:
                pose = self.pose_est.track_one(rgb=color, depth=depth, K=self.cam_K, iteration=2)
                if pose is None:
                    self.get_logger().warn("Lost tracking")
                    self.initialized = False
                    self.selected_mask = None
                    self.preview_mask = None
                    self.waiting_for_selection = False
            else:
                pose = self.pose_est.register(K=self.cam_K, rgb=color, depth=depth, 
                                            ob_mask=self.object_mask, iteration=4)

            if pose is not None:
                # Store the pose for next frame's motion compensation
                self.pose_est.last_pose = pose
                
                # Transform pose to world coordinates
                world_pose = current_camera_pose @ pose @ np.linalg.inv(self.bound)
                self.publish_pose(world_pose)
                
                vis_img = color.copy()
                if self.object_mask is not None and self.mask_color is not None:
                    mask_overlay = np.zeros_like(vis_img)
                    mask_overlay[self.object_mask > 0] = self.mask_color
                    cv2.addWeighted(vis_img, 1, mask_overlay, 0.3, 0, vis_img)
                
                # Visualize pose in camera coordinates for display
                vis_img = self.visualize_pose(vis_img, pose @ np.linalg.inv(self.bound))
                
                position = world_pose[:3, 3]
                rotation = R.from_matrix(world_pose[:3, :3]).as_euler('xyz', degrees=True)
                
                info_text = [
                    f"Tracking: {self.target_class_name}",
                    f"World Position (x,y,z): {position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}",
                    f"World Rotation (deg): {rotation[0]:.1f}, {rotation[1]:.1f}, {rotation[2]:.1f}"
                ]
                
                for i, text in enumerate(info_text):
                    cv2.putText(vis_img, text, (10, 30 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 
                               1, (0, 255, 0), 2, cv2.LINE_AA)
                
                cv2.imshow('Tracking', vis_img[..., ::-1])
                cv2.waitKey(1)

            # Update last camera pose
            self.last_camera_pose = current_camera_pose

    def publish_pose(self, world_pose):
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = self.base_frame

        position = world_pose[:3, 3]
        quaternion = R.from_matrix(world_pose[:3, :3]).as_quat()

        pose_msg.pose.position.x = position[0]
        pose_msg.pose.position.y = position[1]
        pose_msg.pose.position.z = position[2]
        pose_msg.pose.orientation.x = quaternion[0]
        pose_msg.pose.orientation.y = quaternion[1]
        pose_msg.pose.orientation.z = quaternion[2]
        pose_msg.pose.orientation.w = quaternion[3]

        self.pose_pub.publish(pose_msg)

    def visualize_pose(self, image, center_pose):
        vis = draw_posed_3d_box(self.cam_K, img=image, ob_in_cam=center_pose, bbox=self.bbox)
        vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.1, K=self.cam_K, thickness=3, transparency=0, is_input_rgb=True)
        return vis

def main(args=None):
    rclpy.init(args=args)
    node = PoseEstimationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
