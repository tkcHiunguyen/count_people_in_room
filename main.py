import cv2
import threading
from ultralytics import YOLO
import time
import numpy as np
from collections import defaultdict
import json
from datetime import datetime
import random
import queue

# Load YOLO model
# yolo_model = YOLO("yolov8m-finetuned.pt")  # YOLOv8 model
yolo_model = YOLO("yolo11n.pt")
# RTSP camera source
rtsp_url = "rtsp://admin:EYIHXN@192.168.1.19:554/H.264"
video_url = "17555123722.mp4"
cam = 0
cap = cv2.VideoCapture(video_url)

id_map = {}
# Shared variables
latest_frame = None
processing_done = True  # Flag to indicate if processing is finished
stop_threads = False  # Flag to stop threads


undefined = "undefined"
inside = "inside"
outside = "outside"
disappear = "disappear"
last_status = "last_status"
now_status = "now_status"
appear_init = "appear_init"
time_start = "time_start"
time_last = "time_last"
tracking_storage = defaultdict(
    lambda: {
        last_status: "",
        now_status: "",
        appear_init: "",
        time_start: "",
        time_last: "",
    }
)


height = 0
width = 0


def is_inside_box(center_x, center_y, rotated_box_points):
    """
    Kiểm tra xem điểm (center_x, center_y) có nằm trong hình chữ nhật xoay hay không.
    """
    result = cv2.pointPolygonTest(rotated_box_points, (center_x, center_y), False)
    return result >= 0


def capture_camera():
    """Thread to capture frames from the camera."""
    global latest_frame, stop_threads
    while not stop_threads:
        ret, frame = cap.read()
        if not ret:
            print("Error when opening camera")
            break

        frame_height, frame_width = frame.shape[:2]

        # Tính toán chiều rộng và chiều cao của hình chữ nhật theo tỷ lệ phần trăm
        rect_width = int(frame_width * 0.18)  # 18% chiều rộng
        rect_height = int(frame_height * 0.7)  # 70% chiều cao

        # Tính toán góc trên bên trái của hình chữ nhật
        top_left = (
            frame_width - rect_width - int(frame_width * 0.35),
            frame_height - rect_height - int(frame_height * 0.12),
        )

        # Góc dưới bên phải của hình chữ nhật
        bottom_right = (top_left[0] + rect_width, top_left[1] + rect_height)

        # Tính toán tâm của hình chữ nhật để thực hiện quay
        center = (top_left[0] + rect_width // 2, top_left[1] + rect_height // 2)

        # Ma trận quay: 4 độ theo chiều kim đồng hồ
        rotation_matrix = cv2.getRotationMatrix2D(center, -4, 1)

        # Các điểm của hình chữ nhật
        rect_points = np.array(
            [
                top_left,
                (top_left[0] + rect_width, top_left[1]),  # Góc trên bên phải
                bottom_right,  # Góc dưới bên phải
                (top_left[0], top_left[1] + rect_height),  # Góc dưới bên trái
            ],
            dtype=np.float32,
        )

        # Quay các điểm của hình chữ nhật
        rotated_points = cv2.transform(np.array([rect_points]), rotation_matrix)[0]

        # Vẽ hình chữ nhật đã quay bằng cách nối các điểm đã quay
        rotated_points = np.int32(rotated_points)
        color = (153, 0, 0)  # Màu đỏ (BGR)
        thickness = 2  # Độ dày của đường viền
        cv2.polylines(
            frame, [rotated_points], isClosed=True, color=color, thickness=thickness
        )
        global top_left_door
        global top_right_door
        global bottom_right_door
        global bottom_left_door

        top_left_door = rotated_points[0]  # Đỉnh trên bên trái
        top_right_door = rotated_points[1]  # Đỉnh trên bên phải
        bottom_right_door = rotated_points[2]  # Đỉnh dưới bên phải
        bottom_left_door = rotated_points[3]  # Đỉnh dưới bên trái
        # global counting_in
        # global counting_out
        # countin=counting_in| 0
        # countout=counting_out| 0

        # Cập nhật frame mới
        latest_frame = frame.copy()
        fps = cap.get(cv2.CAP_PROP_FPS)
        time.sleep(1 / fps)


def process_tracking():
    """Thread to process YOLO tracking."""
    global latest_frame, processing_done, stop_threads

    # Chờ cho đến khi latest_frame được cập nhật
    while latest_frame is None and not stop_threads:
        print("waiting for frame...")
        time.sleep(0.1)  # Chờ 100ms

    # Khởi tạo VideoWriter (sau khi có kích thước frame)
    output_file = "output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 20
    height, width, channels = latest_frame.shape
    frame_size = (width, height)

    video_writer = cv2.VideoWriter(
        filename=output_file,
        fourcc=fourcc,
        fps=fps,
        frameSize=  # The above code is defining a
        # variable `frame_size` in Python.
        frame_size,
    )
    global counting_in
    global counting_out
    counting_in = 0
    counting_out = 0
    while not stop_threads:
        if latest_frame is not None and processing_done:
            processing_done = False  # Block processing until this is finished
            # YOLOv8 Tracking
            # results = yolo_model.track(
            #     source=latest_frame, tracker="bytetrack.yaml", show=False, save=False, persist=True
            # )

            results = yolo_model.track(
                source=latest_frame, show=False, save=False, persist=True, classes=0
            )
            # print(tracking_storage)
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xywh.cpu()  # Bounding boxes (xywh format)
                track_ids = results[0].boxes.id.cpu().numpy()  # Track IDs
                confidences = results[0].boxes.conf.cpu().numpy()  # Confidence scores
                for box, track_id, conf in zip(boxes, track_ids, confidences):
                    if conf > 0.7:  # Ngưỡng độ tin cậy

                        x, y, w, h = box
                        x, y, w, h = int(x), int(y), int(w), int(h)
                        track_id = int(track_id)
                        # Vẽ bounding box gốc (màu xanh)
                        cv2.rectangle(
                            latest_frame,
                            (x - w // 2, y - h // 2),  # Top left corner
                            (x + w // 2, y + h // 2),  # Bottom right
                            (0, 255, 0),
                            2,  # Green, thickness = 2
                        )
                        # Hiển thị track ID trên bounding box
                        # Tính toán kích thước và vị trí của bounding box màu cam
                        orange_box_w = int(
                            w * 0.3
                        )  # Chiều rộng là 20% của bounding box mẹ
                        orange_box_h = int(h * 0.3)  # Chiều cao không thay đổi
                        orange_box_x = (
                            x - orange_box_w // 2
                        )  # Căn giữa theo chiều ngang
                        orange_box_y = y - h // 2  # Căn trên 10% theo chiều dọc
                        # Vẽ bounding box màu cam
                        cv2.rectangle(
                            latest_frame,
                            (orange_box_x, orange_box_y),
                            (orange_box_x + orange_box_w, orange_box_y + orange_box_h),
                            (0, 165, 255),  # Màu cam (BGR)
                            2,  # Độ dày của đường viền
                        )
                        center_x = orange_box_x + orange_box_w // 2
                        center_y = orange_box_y + orange_box_h // 2
                        cv2.circle(
                            latest_frame,
                            (center_x, center_y),  # Tọa độ điểm chính giữa
                            5,  # Bán kính của điểm
                            (0, 0, 255),  # Màu đỏ (BGR)
                            -1,  # Điền đầy điểm
                        )

                        # Define the reference (rotated door) as the 4 points
                        reference_points = np.array(
                            [
                                top_left_door,
                                top_right_door,
                                bottom_right_door,
                                bottom_left_door,
                            ],
                            dtype=np.float32,
                        )
                        # current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        if track_id not in tracking_storage:
                            # tracking_storage[track_id]=track_id
                            tracking_storage[track_id][last_status] = undefined
                            tracking_storage[track_id][now_status] = undefined
                            tracking_storage[track_id][appear_init] = undefined
                            tracking_storage[track_id][time_last] = "NO"
                            tracking_storage[track_id][time_start] = time.ctime(
                                time.time()
                            )
                            

                        # init where human appear
                        if tracking_storage[track_id][
                                appear_init
                            ] == undefined and is_inside_box(
                                center_x=center_x,
                                center_y=center_y,
                                rotated_box_points=reference_points,
                            ):
                                tracking_storage[track_id][appear_init] = inside
                        else:
                                tracking_storage[track_id][appear_init] = outside
                                
                        # Kiểm tra trạng thái hiện tại (inside hay outside)
                        current_status = "inside" if is_inside_box(
                            center_x=center_x,
                            center_y=center_y,
                            rotated_box_points=reference_points,
                        ) else "outside"

                        # Nếu now_status khác với trạng thái hiện tại, cập nhật last_status và now_status
                        if tracking_storage[track_id][now_status] != current_status:
                            tracking_storage[track_id][last_status] = tracking_storage[track_id][now_status]
                            tracking_storage[track_id][now_status] = current_status
                            


                        info = f"ID: {track_id} {tracking_storage[track_id][now_status]} Conf: {conf:.2f}"
                        text_size = cv2.getTextSize(
                            info, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                        )[0]
                        text_w, text_h = text_size
                        cv2.rectangle(
                            latest_frame,
                            (
                                x - w // 2,
                                y - h // 2 - 30,
                            ),  # Vị trí của nền (phía trên bbox)
                            (
                                x - w // 2 + text_w + 10,
                                y - h // 2 - 30 + text_h + 10,
                            ),  # Kích thước nền
                            (0, 255, 255),  # Màu vàng (BGR)
                            -1,  # Điền đầy nền
                        )
                        cv2.putText(
                            latest_frame,
                            info,
                            (x - w // 2 + 5, y - h // 2 - 5),  # Vị trí của chữ trên nền
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,  # Kích thước chữ
                            (0, 0, 0),  # Màu chữ đen
                            2,  # Độ dày chữ
                        )
            else:
                track_ids_to_delete = []
                for track_id in list(tracking_storage.keys()):
                    current_data = tracking_storage[track_id]  
                    # Kiểm tra điều kiện người đi vào (count in)
                    if (
                        # current_data[appear_init] == outside
                        current_data[now_status] == inside
                        and current_data[time_last] == "NO"
                    ):
                        print("start in......")
                        counting_in += 1
                        current_data[time_last] = "YES"
                        print(f"Person {track_id} counted IN.")
                        
                    
                    # Kiểm tra điều kiện người đi ra (count out)
                    if (
                        # current_data[appear_init] == inside
                        current_data[now_status] == outside
                        and current_data[time_last] == "NO"
                    ):  
                        print("start out.............")
                        counting_out += 1
                        counting_in -= 1
                        current_data[time_last] = "YES"
                        print(f"Person {track_id} counted OUT.")
                        # del tracking_storage[track_id]  # Xóa track_id khỏi storage

                    print(f"Tracking ID {track_id}: {current_data}")
                    if current_data[time_last] == "YES":
                        track_ids_to_delete.append(track_id)
                for track_id in track_ids_to_delete:
                    del tracking_storage[track_id]
                    print(f"Deleted track_id: {track_id}")

            
            in_text = f"IN: {counting_in}"
            out_text = f"OUT: {counting_out}"
            # Kích thước chữ và độ dày
            font_scale = 1.2  # Cỡ chữ
            thickness = 4  # Độ dày chữ
            # Tính kích thước của nội dung "IN" và "OUT"
            in_text_size = cv2.getTextSize(
                in_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )[0]
            out_text_size = cv2.getTextSize(
                out_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )[0]
            # Vị trí hiển thị (góc phải bên trên)
            in_x = width - in_text_size[0] - 20  # Cách lề phải 20 pixel
            in_y = 50  # Cách lề trên 50 pixel
            out_x = width - out_text_size[0] - 20
            out_y = in_y + in_text_size[1] + 20  # Dưới "IN" 20 pixel
            # Vẽ nền cho "IN"
            cv2.rectangle(
                latest_frame,
                (in_x - 10, in_y - in_text_size[1] - 10),  # Top left
                (in_x + in_text_size[0] + 10, in_y + 10),  # Bottom right
                (255, 255, 255),  # Màu nền trắng
                -1,  # Điền đầy
            )
            # Vẽ chữ "IN"
            cv2.putText(
                latest_frame,
                in_text,
                (in_x, in_y),  # Vị trí của chữ
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 0),  # Màu xanh cyan (BGR)
                thickness,
            )
            # Vẽ nền cho "OUT"
            cv2.rectangle(
                latest_frame,
                (out_x - 10, out_y - out_text_size[1] - 10),  # Top left
                (out_x + out_text_size[0] + 10, out_y + 10),  # Bottom right
                (255, 255, 255),  # Màu nền trắng
                -1,  # Điền đầy
            )
            # Vẽ chữ "OUT"
            cv2.putText(
                latest_frame,
                out_text,
                (out_x, out_y),  # Vị trí của chữ
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 165, 255),  # Màu cam (BGR)
                thickness,
            )
            processing_done = True
            video_writer.write(latest_frame)
            # Hiển thị frame trong thời gian thực (tuỳ chọn)
            cv2.imshow("Processed Frame", latest_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                stop_threads = True
                break
    video_writer.release()
    cv2.destroyAllWindows()


# Start threads
camera_thread = threading.Thread(target=capture_camera, daemon=True)
tracking_thread = threading.Thread(target=process_tracking, daemon=True)

camera_thread.start()
tracking_thread.start()

try:
    while True:
        time.sleep(1)  # Keep the main thread alive
except KeyboardInterrupt:
    stop_threads = True
    camera_thread.join()
    tracking_thread.join()
    cap.release()
    print("Program terminated.")
