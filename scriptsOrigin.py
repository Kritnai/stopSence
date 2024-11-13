# my predict
import supervision as sv
from ultralytics import YOLO
import cv2

import numpy as np
from collections import defaultdict, deque


media = "vehicles.mp4"
mediaOutput = "vehiclesOutput.mp4"


SOURCE = np.array([
    [1252, 787],
    [2298, 803],
    [5039, 2159],
    [-550, 2159]
])

TARGET_WIDTH = 25
TARGET_HIGH = 250

TARGET = np.array([
    [0, 0],
    [TARGET_WIDTH - 1, 0],
    [TARGET_WIDTH - 1, TARGET_HIGH - 1],
    [0, TARGET_HIGH - 1]
])

class ViewsTranformer:
    def __init__(self, source: np.ndarray, target: np.ndarray ):
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m =  cv2.getPerspectiveTransform(source, target)

    def transform_point(self, point: np.ndarray) -> np.ndarray:
        reshaped_point = point.reshape(-1, 1, 2).astype(np.float32)
         
        transform_points = cv2.perspectiveTransform(reshaped_point, self.m)
        return transform_points.reshape(-1, 2)
         
    # def transform_point(self, point: np.ndarray) -> np.ndarray:
    #     # Ensure the input point is in the correct shape for perspectiveTransform
    #     if len(point.shape) == 2 and point.shape[1] == 2:
    #         reshaped_point = point.reshape(-1, 1, 2).astype(np.float32)
    #     else:
    #         raise ValueError("Input point must have shape (N, 2) for transformation.")
        
    #     transform_points = cv2.perspectiveTransform(reshaped_point, self.m)
    #     return transform_points.reshape(-1, 2)



if __name__ == "__main__":


    video_info = sv.VideoInfo.from_video_path(f"data/{media}")
    model = YOLO("yolov8x.pt")

    byte_track = sv.ByteTrack(frame_rate=video_info.fps )

 
    
    # color_lookup=sv.ColorLookup.TRACK # เป็นการทำให้แต่ละ object ใน frame มีสีที่ต่างกัน
    
    # ปรับขนาดของเส้นและตัวอักษร
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=video_info.resolution_wh)
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh)
    
    # bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=thickness)
    
    # color of rectangle
    bounding_box_annotator = sv.BoxAnnotator(thickness=thickness, color_lookup=sv.ColorLookup.TRACK) 
    # color of text backgroud
    label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_thickness=thickness, text_position=sv.Position.BOTTOM_CENTER, color_lookup=sv.ColorLookup.TRACK)
    # color of line follow
    trace_annotor = sv.TraceAnnotator(thickness=thickness, trace_length=video_info.fps * 2, position=sv.Position.BOTTOM_CENTER, color_lookup=sv.ColorLookup.TRACK)
    
 
    # อ่านภาพจากคลิปทีละ frame
    frame_generator = sv.get_video_frames_generator(f"data/{media}")

    # polygon_zone = sv.PolygonZone(SOURCE, frame_resolution_wh=video_info.resolution_wh) # แบบเก่า ไม่ได้ใช้
    polygon_zone = sv.PolygonZone(SOURCE)


    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None

    # # ดุูความคืบหน้าของการ predict
    # currentFrame = 0
    # allFrame = len(annotated_frame)

    # ปรับค่าที่ได้ให้เป็น perspective
    views_tranformer =  ViewsTranformer(source=SOURCE, target=TARGET)

    # ใช้สำหรับคิดความเร็ว ยังไม่รู้คือไร
    coordinate = defaultdict(lambda: deque(maxlen=video_info.fps))
    for frame in frame_generator:
        # # Iterate over frames and check for None
        if frame is None:
            print("Failed to read frame. Check the video file or generator.")
            break
        # # Initialize VideoWriter with the correct frame size
        if out is None:
            height, width, _ = frame.shape
            out = cv2.VideoWriter(f'data/output/{mediaOutput}', fourcc, 20.0, (width, height))


        result = model(frame)[0]
        detections = sv.Detections.from_ultralytics(result)
        
        # ใส่ zone สำหรับ detect
        detections = detections[polygon_zone.trigger(detections)]

        # เป็นการเพิ่มคำสั่งจำหรับจ object
        detections = byte_track.update_with_detections(detections=detections)

        # หาตำแหน่งของรถ จาก target หรือมุมมอง จริง
        points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        points = views_tranformer.transform_point(point=points).astype(int)
       


        labels = []
        speed = 0
        for tracker_id, [_, y] in zip(detections.tracker_id, points):
            coordinate[tracker_id].append(y)
            # กรณีที่ object มีข้อมูลไม่มากพอคำนวน
            if len(coordinate[tracker_id]) < video_info.fps / 2:
                labels.append(f"#{tracker_id}") 
            
            # กรณีที่ object มีข้อมูลพอคำนวน
            else:
                coordinate_start = coordinate[tracker_id][-1]
                coordinate_end = coordinate[tracker_id][0]
                distance = abs(coordinate_start - coordinate_end)
                time = len(coordinate[tracker_id]) / video_info.fps
                speed = distance / time * 3.6
                labels.append(f"#{tracker_id} {int(speed)} km/h")

        # labels = [
        #     # f"#{tracker_id}"
        #     # for tracker_id
        #     # in detections.tracker_id

        #     f"x: {x}, y: {y}"
        #     for x, y
        #     in points
        # ]

            
        annotated_frame = frame.copy()
        
        # เส้นบอกพื้นที่ตรวจจับ 
        annotated_frame = sv.draw_polygon(annotated_frame, polygon=SOURCE, color=sv.Color.GREEN)
            
            
        annotated_frame = trace_annotor.annotate(
            scene=annotated_frame, detections=detections
        )

        annotated_frame = bounding_box_annotator.annotate(
            scene=annotated_frame, detections=detections
        )
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame, detections=detections, labels=labels
        )

        # # แสดงความคืบหน้าของการ predict
        # print(f"on frame: {currentFrame + 1}/{allFrame}")
        # currentFrame += 1


        # not work for jupiter notebook but work for file.py
        # cv2.imshow("annotated_frame", annotated_frame)
        # if cv2.waitKey(1) == ord("q"):
        #     break


        # Write the annotated frame to the output video
        out.write(annotated_frame)



    # Release the VideoWriter and close all windows
    out.release()
    cv2.destroyAllWindows()
