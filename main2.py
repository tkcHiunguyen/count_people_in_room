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


stop_threads = False
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
flag = "flag"
exs = "exs"
tracking_storage = defaultdict(
    lambda: {
        last_status: "",
        now_status: "",
        appear_init: "",
        flag: "",
        exs: 0,
    }
)
list_in = []


height = 0
width = 0


def is_inside_box(center_x, center_y, rotated_box_points):
    """
    Kiểm tra xem điểm (center_x, center_y) có nằm trong hình chữ nhật xoay hay không.
    """
    result = cv2.pointPolygonTest(rotated_box_points, (center_x, center_y), False)
    return result >= 0
def capture_camera():
    global latest_frame, top_left_door, top_right_door, bottom_right_door, bottom_left_door
    while not stop_threads:
        ret, frame = cap.read()
        if not ret:
            print("Error when opening camera")
            break

        frame_height, frame_width = frame.shape[:2]

        # Tính toán kích thước hình chữ nhật (18% chiều rộng, 70% chiều cao)
        rect_width = int(frame_width * 0.18)
        rect_height = int(frame_height * 0.7)

        # Tính toán tọa độ góc trên bên trái của hình chữ nhật
        top_left = (
            frame_width - rect_width - int(frame_width * 0.35),
            frame_height - rect_height - int(frame_height * 0.12),
        )

        # Góc dưới bên phải của hình chữ nhật
        bottom_right = (top_left[0] + rect_width, top_left[1] + rect_height)

        # Tính toán tâm hình chữ nhật để quay
        center = (top_left[0] + rect_width // 2, top_left[1] + rect_height // 2)

        # Ma trận quay (quay -4 độ theo chiều kim đồng hồ)
        rotation_matrix = cv2.getRotationMatrix2D(center, -4, 1)

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
        # Chuyển các điểm đã quay thành kiểu int và vẽ lên khung hình
        rotated_points = np.int32(rotated_points)
        color = (153, 0, 0)  # Màu đỏ (BGR)
        thickness = 2  # Độ dày của đường viền
        cv2.polylines(
            frame, [rotated_points], isClosed=True, color=color, thickness=thickness
        )
        # Lưu lại các điểm của hình chữ nhật đã quay vào các biến toàn cục
        top_left_door = rotated_points[0]  # Đỉnh trên bên trái
        top_right_door = rotated_points[1]  # Đỉnh trên bên phải
        bottom_right_door = rotated_points[2]  # Đỉnh dưới bên phải
        bottom_left_door = rotated_points[3]  # Đỉnh dưới bên trái
        latest_frame = frame.copy()
        fps = cap.get(cv2.CAP_PROP_FPS)
        time.sleep(1 / fps)
def process_tracking():
    global latest_frame, processing_done, stop_threads, list_in, tracking_storage
    while latest_frame is None and not stop_threads:
        print("waiting for frame...")
        time.sleep(0.1)  # Chờ 100ms
    fps = 20  # Cần lấy FPS từ video gốc nếu có
    height, width, channels = latest_frame.shape
    frame_size = (width, height)
    output_file = "output2.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Hỗ trợ định dạng mp4
    video_writer = cv2.VideoWriter(
        filename=output_file,
        fourcc=fourcc,
        fps=fps,
        frameSize=frame_size,
    )
    counting_in = 0
    counting_out = 0
    while not stop_threads:
        if latest_frame is not None and processing_done:
            processing_done = False
            frame = latest_frame.copy()
            results = yolo_model.track(
                source=frame, show=False, save=False, persist=True, classes=0
            )
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xywh.cpu().numpy()  # Các bounding boxes
                track_ids = results[0].boxes.id.cpu().numpy()  # Các ID đối tượng
                confidences = results[0].boxes.conf.cpu().numpy()  # Điểm tin cậy
                for i in range(len(boxes)):
                    x, y, w, h = boxes[i]
                    track_id = track_ids[i]
                    confidence = confidences[i]
                    x, y, w, h = int(x), int(y), int(w), int(h)
                    track_id = int(track_id)
                    cv2.rectangle(
                        frame,
                        (x - w // 2, y - h // 2),  # Top left corner
                        (x + w // 2, y + h // 2),  # Bottom right
                        (0, 255, 0),
                        2,
                    )
                    # Tính toán kích thước và vị trí của bounding box màu cam
                    orange_box_w = int(w * 0.3)  # Chiều rộng là 20% của bounding box mẹ
                    orange_box_h = int(h * 0.3)  # Chiều cao không thay đổi
                    orange_box_x = x - orange_box_w // 2  # Căn giữa theo chiều ngang
                    orange_box_y = y - h // 2  # Căn trên 10% theo chiều dọc
                    # Vẽ bounding box màu cam
                    cv2.rectangle(
                        frame,
                        (orange_box_x, orange_box_y),
                        (orange_box_x + orange_box_w, orange_box_y + orange_box_h),
                        (0, 165, 255),  # Màu cam (BGR)
                        2,  # Độ dày của đường viền
                    )
                    center_x = orange_box_x + orange_box_w // 2
                    center_y = orange_box_y + orange_box_h // 2
                    cv2.circle(
                        frame,
                        (center_x, center_y),  # Tọa độ điểm chính giữa
                        5,  # Bán kính của điểm
                        (0, 0, 255),  # Màu đỏ (BGR)
                        -1,  # Điền đầy điểm
                    )
                    reference_points = np.array(
                        [
                            top_left_door,
                            top_right_door,
                            bottom_right_door,
                            bottom_left_door,
                        ],
                        dtype=np.float32,
                    )

                    if track_id not in tracking_storage:
                        tracking_storage[track_id] = {
                            last_status: undefined,
                            now_status: undefined,
                            appear_init: undefined,
                            flag: "NO",
                            exs: 0,
                        }
                    if tracking_storage[track_id][appear_init] == undefined:
                        if is_inside_box(
                            center_x, center_y, rotated_box_points=reference_points
                        ):
                            tracking_storage[track_id][appear_init] = inside
                        else:
                            tracking_storage[track_id][appear_init] = outside
                    if is_inside_box(
                        center_x, center_y, rotated_box_points=reference_points
                    ):
                        if tracking_storage[track_id][now_status] != inside:
                            tracking_storage[track_id][last_status] = tracking_storage[
                                track_id
                            ][now_status]
                            tracking_storage[track_id][now_status] = inside
                    else:
                        if tracking_storage[track_id][now_status] != outside:
                            tracking_storage[track_id][last_status] = tracking_storage[
                                track_id
                            ][now_status]
                            tracking_storage[track_id][now_status] = outside
                    tracking_storage[track_id][exs] += 1
                    # info = f"ID: {track_id} {tracking_storage[track_id][now_status]} {tracking_storage[track_id][last_status]}"
                    info = f"ID: {track_id} {tracking_storage[track_id][now_status]}"
                    text_size = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[
                        0
                    ]
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
                for track_id in list(tracking_storage.keys()):
                    print(f"{tracking_storage[track_id]}-{tracking_storage[track_id][exs]} ")
                    if tracking_storage[track_id][exs]<15 or tracking_storage[track_id][now_status]==tracking_storage[track_id][appear_init]==outside:
                        del tracking_storage[track_id]
                for track_id in list(tracking_storage.keys()):
                    obj = tracking_storage[track_id]
                    if obj[exs] >= 15:
                        if (
                            obj[now_status] == inside
                            and obj[flag] == "NO"
                            and obj[last_status] == outside
                        ):
                            counting_in += 1
                            obj[flag] = "YES"
                        if (
                            obj[now_status] == outside
                            and obj[appear_init] == inside
                            and obj[flag] == "NO"
                        ):
                            counting_in -= 1
                            obj[flag] = "YES"
            for track_id in list(tracking_storage.keys()):
                obj = tracking_storage[track_id]
                if obj[flag] == "YES":
                    del tracking_storage[track_id]
                    print(f"Deleted track_id {track_id} from tracking_storage")
            processing_done = True
            in_text = f"IN: {counting_in}"
            out_text = f"OUT: {counting_out}"
            font_scale = 1.2  # Cỡ chữ
            thickness = 4  # Độ dày chữ
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
                frame,
                (in_x - 10, in_y - in_text_size[1] - 10),  # Top left
                (in_x + in_text_size[0] + 10, in_y + 10),  # Bottom right
                (255, 255, 255),  # Màu nền trắng
                -1,  # Điền đầy
            )
            cv2.putText(
                frame,
                in_text,
                (in_x, in_y),  # Vị trí của chữ
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 0),  # Màu xanh cyan (BGR)
                thickness,
            )
            # cv2.rectangle(
            #     frame,
            #     (out_x - 10, out_y - out_text_size[1] - 10),  # Top left
            #     (out_x + out_text_size[0] + 10, out_y + 10),  # Bottom right
            #     (255, 255, 255),  # Màu nền trắng
            #     -1,
            # )
            # cv2.putText(
            #     frame,
            #     out_text,
            #     (out_x, out_y),  # Vị trí của chữ
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     font_scale,
            #     (0, 165, 255),  # Màu cam (BGR)
            #     thickness,
            # )
            ################################################
            ##          AIR CONDITION                    ###
            ################################################
            if counting_in==0:
                text_color=(255,255,255)
                background_color=(0,0,255)
            else:
                background_color = (255, 255, 0)  # Màu cyan (BGR)
                text_color = (0, 0, 0)  # Màu trắng
            rect_width = int(width * 0.2)  # 10% chiều rộng
            rect_height = int(width * 0.07)  # 5% chiều rộng
            top_left = (0, 0) 
            bottom_right = (top_left[0] + rect_width, top_left[1] + rect_height)
            cv2.rectangle(frame, top_left, bottom_right, background_color, -1)
            text_position = (top_left[0] + 10, top_left[1] + rect_height // 2 + 10)
            
            cv2.putText(
                frame,
                "AIR CONDITION",
                text_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,  # Kích thước font
                text_color,
                2,  # Độ dày nét chữ
                cv2.LINE_AA,
            )
            
            video_writer.write(frame)
            cv2.imshow("Processed Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                stop_threads = True
                break

    video_writer.release()
    cv2.destroyAllWindows()


def main():
    global stop_threads

    camera_thread = threading.Thread(target=capture_camera, name="CaptureCameraThread")
    tracking_thread = threading.Thread(
        target=process_tracking, name="ProcessTrackingThread"
    )

    camera_thread.start()
    tracking_thread.start()

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt detected! Stopping threads...")
        stop_threads = True

        camera_thread.join()
        tracking_thread.join()
        print("All threads stopped. Exiting program.")


if __name__ == "__main__":
    main()
