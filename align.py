import pyrealsense2 as rs
import numpy as np
import cv2

import argparse

class Alignment:
    def __init__(self, clipping_distance_in_meters = 1, fps = 30):
        
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        self.pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        self.pipeline_profile = self.config.resolve(self.pipeline_wrapper)
        self.device = self.pipeline_profile.get_device()
        self.device_product_line = str(self.device.get_info(rs.camera_info.product_line))

        found_rgb = False  
        for s in self.device.sensors: 
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            print("The demo requires Depth camera with Color sensor")
            exit(0)

        self.fps = fps

        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, self.fps)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, self.fps)

        # Start streaming
        self.profile = self.pipeline.start(self.config)

        # Getting the depth sensor's depth scale (see rs-align example for explanation)
        self.depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = self.depth_sensor.get_depth_scale()
        #print("Depth Scale is: " , self.depth_scale)

        #We will be removing  the background of objects more than
        #clipping_distance_in_meters meters away
        self.clipping_distance_in_meters = clipping_distance_in_meters #1 meter
        self.clipping_distance = self.clipping_distance_in_meters / self.depth_scale

        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.
        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)

    def aligned_frames(self):
        # Get frameset of color and depth
        frames = self.pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = self.align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        #   Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            return None, None            

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        return depth_image, color_image
        
    def remove_bg(self, depth_image, color_image, bg_color = 153):
        # Remove background - Set pixels further than clipping_distance to grey
            
        depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
        bg_removed = np.where((depth_image_3d > self.clipping_distance) | (depth_image_3d <= 0), bg_color, color_image)

        return bg_removed
        
    def render(self, bg_removed, depth_image):
        # Render images:
        #depth align to color on left
        #depth on right
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        images = np.hstack((bg_removed, depth_colormap))
        return images
    
    def hsv_mask(self, color_image, low_H=0, low_S=0, low_V=0, high_H=180, high_S=255, high_V=255):
        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (low_H, low_S, low_V), (high_H, high_S, high_V))
        masked_image = cv2.bitwise_and(color_image, color_image, mask=mask)
        return mask, masked_image
        
    def stop(self):
        self.pipeline.stop()

max_value = 255
max_value_H = 180

# Initial HSV thresholds
low_H, low_S, low_V = 0, 0, 0
high_H, high_S, high_V = 180, 255, 255

# Window names
window_capture_name = 'Video Capture'
window_detection_name = 'Object Detection'

# Trackbar callbacks
def nothing(x):
    pass

if __name__ == "__main__":
    
    # Initial HSV thresholds
    max_value = 255
    max_value_H = 180
    low_H, low_S, low_V = 0, 0, 0
    high_H, high_S, high_V = 180, 255, 255

    # Single window for display + trackbars
    window_name = 'Align Example'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Trackbars
    cv2.createTrackbar('Low H', window_name, low_H, max_value_H, nothing)
    cv2.createTrackbar('High H', window_name, high_H, max_value_H, nothing)
    cv2.createTrackbar('Low S', window_name, low_S, max_value, nothing)
    cv2.createTrackbar('High S', window_name, high_S, max_value, nothing)
    cv2.createTrackbar('Low V', window_name, low_V, max_value, nothing)
    cv2.createTrackbar('High V', window_name, high_V, max_value, nothing)

    align = Alignment(clipping_distance_in_meters=1, fps=30)

    try:
        while True:
            depth_image, color_image = align.aligned_frames()
            if depth_image is None or color_image is None:
                continue

            # Remove background
            bg_removed = align.remove_bg(depth_image, color_image, bg_color=153)

            # Read HSV values from trackbars
            low_H = cv2.getTrackbarPos('Low H', window_name)
            high_H = cv2.getTrackbarPos('High H', window_name)
            low_S = cv2.getTrackbarPos('Low S', window_name)
            high_S = cv2.getTrackbarPos('High S', window_name)
            low_V = cv2.getTrackbarPos('Low V', window_name)
            high_V = cv2.getTrackbarPos('High V', window_name)

            # Apply HSV mask
            mask, masked_image = align.hsv_mask(bg_removed, low_H, low_S, low_V,
                                                high_H, high_S, high_V)

            # Render masked image + depth
            images = align.render(masked_image, depth_image)

            # Show in single window
            cv2.imshow(window_name, images)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:
                break
    finally:
        align.stop()
        cv2.destroyAllWindows()