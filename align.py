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
    
    def hsv_mask(self, color_image, lower_hsv = np.array([0, 0, 0]), higher_hsv = np.array([255, 255, 255])):
        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_hsv, higher_hsv)
        return mask
    
    def add_contour(self,mask=None):
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        return contours

        
    def render(self, color_image, depth_image, mask=None, contours = None):
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        if mask is not None:
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            #images = np.hstack((bg_removed, mask_bgr, depth_colormap))
            pen_only = cv2.bitwise_and(color_image, color_image, mask=mask)

            if contours is not None and len(contours) > 0:
                
                cv2.drawContours(pen_only, contours, -1, (0, 255, 0), 2)
            
            images = np.hstack((color_image, mask_bgr, pen_only))
        else:
            images = np.hstack((bg_removed, depth_colormap))

        return images      
    
    
        
    def stop(self):
        self.pipeline.stop()



lower_hsv = np.array([111, 95, 89])
upper_hsv = np.array([135, 255, 174])

# Window names
#window_capture_name = 'Video Capture'
#window_detection_name = 'Object Detection'

# Trackbar callbacks
def nothing(x):
    pass

if __name__ == "__main__":

    import cv2
    import numpy as np

    # BGR value (since OpenCV reads in BGR, not RGB)
    #bgr = np.uint8([[[62, 62, 129]]])
    #hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    #print(hsv)    

    # Single window for display + trackbars
    window_name = 'Align Example'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    

    align = Alignment(clipping_distance_in_meters=1, fps=30)

    try:
        while True:
            depth_image, color_image = align.aligned_frames()
            if depth_image is None or color_image is None:
                continue

            # Remove background
            bg_removed = align.remove_bg(depth_image, color_image, bg_color=153)          

            # Apply HSV mask
            mask = align.hsv_mask(color_image, lower_hsv=lower_hsv, higher_hsv=upper_hsv)        

            # Combine background + pen overlay
            #result = cv2.add(img_bg, pen_overlay)

            #cv2.imshow('Pen Overlay', result)

            contour = align.add_contour(mask)
            
            # Render masked image + depth
            images = align.render(color_image, depth_image, mask, contour)

            # Show in single window
            cv2.imshow(window_name, images)
            

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:
                break
    finally:
        align.stop()
        cv2.destroyAllWindows()