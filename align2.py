import pyrealsense2 as rs
import numpy as np
import cv2

class Alignment:
    def __init__(self, clipping_distance_in_meters=1, fps=30):
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Configure streams
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, fps)

        # Start streaming
        self.profile = self.pipeline.start(self.config)

        # Depth scale
        self.depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = self.depth_sensor.get_depth_scale()

        # Clipping distance
        self.clipping_distance_in_meters = clipping_distance_in_meters
        self.clipping_distance = self.clipping_distance_in_meters / self.depth_scale

        # Alignment object
        self.align = rs.align(rs.stream.color)
        self.depth_stream = self.profile.get_stream(rs.stream.depth).as_video_stream_profile()
        self.intrinsics = self.depth_stream.get_intrinsics()

        # HSV range for pen detection
        self.lower_hsv = np.array([111, 95, 89])
        self.upper_hsv = np.array([135, 255, 174])

    def update(self):
        # Get frames
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            return None, None

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Remove background
        depth_image_3d = np.dstack((depth_image, depth_image, depth_image))
        bg_removed = np.where((depth_image_3d > self.clipping_distance) | (depth_image_3d <= 0), 153, color_image)

        # HSV mask
        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)

        # Morphological closing
        kernel = np.ones((7,7), np.uint8)
        mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Render
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        pen_only = cv2.bitwise_and(bg_removed, bg_removed, mask=mask)
        pen_blur = cv2.blur(pen_only, (10,10))

        if contours and len(contours) > 0:
            cnt = max(contours, key=cv2.contourArea)
            if len(cnt) >= 5:
                ellipse = cv2.fitEllipse(cnt)
                cv2.ellipse(pen_blur, ellipse, (0,255,0), 2)
                (cx, cy), _, _ = ellipse

                # Clamp coordinates
                h, w = depth_image.shape
                px = int(max(0, min(cx, w-1)))
                py = int(max(0, min(cy, h-1)))

                # Draw centroid
                cv2.circle(pen_blur, (px, py), 5, (0,0,255), -1)

                # Get 3D point
                depth_value = depth_image[py, px]
                depth_in_meters = depth_value * self.depth_scale
                point_3d = rs.rs2_deproject_pixel_to_point(self.intrinsics, [px, py], depth_in_meters)

                images = np.hstack((pen_blur, bg_removed, depth_colormap))
                return point_3d, images

        images = np.hstack((pen_blur, bg_removed, depth_colormap))
        return None, images

    def stop(self):
        self.pipeline.stop()


if __name__ == "__main__":
    window_name = 'Align'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    align = Alignment(clipping_distance_in_meters=1, fps=30)

    points_history = []

    try:
        while True:
            point_3d, images = align.update()
            if images is not None:
                cv2.imshow(window_name, images)

            if point_3d is not None:
                points_history.append(point_3d)
                print("Current 3D point:", point_3d)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if len(points_history) >= 50:  # Stop after collecting 50 points
                break

    finally:
        if points_history:
            mean_point = np.mean(points_history, axis=0)
            print("Mean 3D point:", mean_point)
        align.stop()
        cv2.destroyAllWindows()
