import cv2
import torch
import numpy as np

import os


# CONGFIG
videos_dir_path = "data/videos"
frame_outputs_dir = "data/frame_outputs"
output_dir = "video_outputs"
fps = 24
alpha = 0.3



def prepare_frames_for_viz(video_path):
    if isinstance(video_path, str) and video_path.endswith(".mp4"):
        cap = cv2.VideoCapture(video_path)
        video_frames_for_vis = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            video_frames_for_vis.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        cap.release()

        return video_frames_for_vis
    else:
        print('Ошибка: путь должен вести к файлу с расширением ".mp4"')
        return


def visualize_frame_cv2(
    frame_idx,
    video_frames,
    outputs_list,
    alpha=0.5
):
    """
    Returns a frame (BGR ndarray) with masks and boxes drawn using cv2.

    Args:
        frame_idx: int – which frame to draw
        video_frames: list of numpy frames (BGR or RGB)
        outputs_list: list of {frame_idx: {obj_id: mask}}
        alpha: mask transparency
    """

    # Ensure list-of-outputs format
    if isinstance(outputs_list, dict):
        outputs_list = [outputs_list]

    frame = video_frames[frame_idx].copy()

    # Convert to BGR consistently
    if frame.shape[2] == 3 and frame.max() <= 1.0:
        frame = (frame * 255).astype(np.uint8)
    img_h, img_w = frame.shape[:2]

    overlay = np.zeros_like(frame, dtype=np.uint8)

    # ================= DRAW MASKS & BOXES =================
    CATEGORY_COLORS = [
    (0, 0, 255),      # RED    - people
    (255, 0, 0),      # BLUE   - vehicles
    (0, 255, 0),      # GREEN  - license plates
    ]

    for cat_idx, outputs_dict in enumerate(outputs_list):
        if frame_idx not in outputs_dict:
            continue

        objects = outputs_dict[frame_idx]
        category_color = CATEGORY_COLORS[cat_idx]

        for obj_id, mask in objects.items():
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()

            if mask.sum() == 0:
                continue

            if mask.shape != (img_h, img_w):
                mask = cv2.resize(mask.astype(np.uint8), (img_w, img_h)) > 0

            overlay[mask] = category_color

            ys, xs = np.where(mask)
            if len(xs) > 0:
                x1, x2 = xs.min(), xs.max()
                y1, y2 = ys.min(), ys.max()
                cv2.rectangle(frame, (x1, y1), (x2, y2), category_color, 2)
                cv2.putText(frame, f"id={obj_id}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, category_color, 2)


        # ======== APPLY MASK TRANSPARENCY ========
        frame = cv2.addWeighted(frame, 1.0, overlay, alpha, 0)

    return frame


if __name__ == '__main__':
    for video_file in os.listdir(videos_dir_path):
        video_name, video_extension = os.path.splitext(video_file)
        video_path = os.path.join(videos_dir_path, video_file)

        outputs_per_frame_people = torch.load(f"{frame_outputs_dir}/{video_name}_outputs_people.pt", weights_only=False)
        outputs_per_frame_vehicle = torch.load(f"{frame_outputs_dir}/{video_name}_outputs_truck.pt", weights_only=False)
        outputs_per_frame_license_plates = torch.load(f"{frame_outputs_dir}/{video_name}_outputs_licenseplate.pt", weights_only=False)

        video_frames_for_vis = prepare_frames_for_viz(video_path)

        cap = cv2.VideoCapture(video_path)
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(f"{output_dir}/{video_name}_outputs.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

        for i in range(len(video_frames_for_vis)):
            frame = visualize_frame_cv2(
                frame_idx=i,
                video_frames=video_frames_for_vis,
                outputs_list=[outputs_per_frame_people, outputs_per_frame_vehicle, outputs_per_frame_license_plates],
                alpha=alpha
            )
            writer.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        writer.release()