import rosbag
import cv2
from cv_bridge import CvBridge

def extract_images(bag_path, output_dir, cameras=5):
    bag = rosbag.Bag(bag_path)
    bridge = CvBridge()
    
    for cam_id in range(cameras):
        topic = f'/camera_{cam_id}/image_raw'
        for idx, (_, msg, _) in enumerate(bag.read_messages(topics=[topic])):
            cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            cv2.imwrite(f"{output_dir}/cam{cam_id}_frame{idx:05d}.png", cv_img)