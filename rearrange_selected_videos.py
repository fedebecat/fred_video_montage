import cv2
import numpy as np
import os
import math

# --- Configuration ---
FRAME_INDEX_TO_LOAD = 49  # 50th frame (index 49)
INPUT_IDS_FILE = 'selected_video_ids.npy'
OUTPUT_PATHS_FILE = 'ordered_frame_paths.npy'
MAX_WINDOW_WIDTH = 1920
MAX_WINDOW_HEIGHT = 1080

# --- State variables for drag-and-drop ---
is_dragging = False
drag_index = -1
drag_image_copy = None
mouse_x, mouse_y = 0, 0

# --- Other global variables ---
frames_data = []
frame_h, frame_w = 0, 0
grid_cols = 0
window_name = 'Frame Sorter (Click for info, Drag to reorder, ENTER to save)'


def extract_timestamp_from_path(path):
    """
    Parses a filename like '.../Video_95_16_02_46.690795.jpg'
    and returns a sortable tuple (hour, minute, second).
    """
    try:
        filename = os.path.basename(path)
        parts = filename.removesuffix('.jpg').split('_')
        hour = int(parts[2])
        minute = int(parts[3])
        second = float(parts[4])
        return (hour, minute, second)
    except (IndexError, ValueError):
        print(f"Warning: Could not parse timestamp from filename: {path}")
        return (99, 99, 99.9)


def on_mouse_event(event, x, y, flags, param):
    """Handles all mouse events for printing info and dragging."""
    global is_dragging, drag_index, drag_image_copy, mouse_x, mouse_y, frames_data

    mouse_x, mouse_y = x, y

    if event == cv2.EVENT_LBUTTONDOWN:
        col = x // frame_w if frame_w > 0 else 0
        row = y // frame_h if frame_h > 0 else 0
        clicked_index = row * grid_cols + col

        if 0 <= clicked_index < len(frames_data):
            # --- NEW: Print info on click ---
            video_id = frames_data[clicked_index]['id']
            path = frames_data[clicked_index]['path']
            h, m, s = extract_timestamp_from_path(path)
            # Nicely format the timestamp string
            timestamp_str = f"{h:02d}:{m:02d}:{s:09.6f}"
            print(f"Clicked Frame -> Video ID: {video_id}, Timestamp: {timestamp_str}")
            # --- End of new code ---

            # Existing drag-and-drop logic starts here
            is_dragging = True
            drag_index = clicked_index
            drag_image_copy = frames_data[drag_index]['image'].copy()
            overlay = np.full(drag_image_copy.shape, (128, 128, 128), dtype=np.uint8)
            drag_image_copy = cv2.addWeighted(drag_image_copy, 0.7, overlay, 0.3, 0)

    elif event == cv2.EVENT_LBUTTONUP:
        if is_dragging:
            col = x // frame_w if frame_w > 0 else 0
            row = y // frame_h if frame_h > 0 else 0
            drop_index = row * grid_cols + col

            if 0 <= drop_index < len(frames_data) and drop_index != drag_index:
                frames_data[drag_index], frames_data[drop_index] = \
                    frames_data[drop_index], frames_data[drag_index]

            is_dragging = False
            drag_index = -1
            drag_image_copy = None


def main():
    global frames_data, frame_h, frame_w, grid_cols

    # 1. Load data
    try:
        selected_ids = np.load(INPUT_IDS_FILE, allow_pickle=True)
        all_paths = np.load('fred_frame_paths.npy', allow_pickle=True).item()
    except FileNotFoundError as e:
        print(f"Error: Could not load a required file: {e.filename}")
        return

    # 2. Populate frames_data list
    print(f"Loading {len(selected_ids)} selected frames...")
    id_set = set(selected_ids)
    for video_id in selected_ids:
        if video_id in all_paths:
            paths_dict = all_paths[video_id]
            if 'RGB' in paths_dict and len(paths_dict['RGB']) > FRAME_INDEX_TO_LOAD:
                path = paths_dict['RGB'][FRAME_INDEX_TO_LOAD]
                if os.path.exists(path):
                    image = cv2.imread(path)
                    if image is not None:
                        frames_data.append({'id': video_id, 'path': path, 'image': image})

    if not frames_data:
        print("No valid frames were loaded. Exiting.")
        return

    # 3. Perform the initial sort
    print("Performing initial sort based on frame timestamps...")
    frames_data.sort(key=lambda item: extract_timestamp_from_path(item['path']))
    print("Sort complete.")

    # 4. Handle resizing to fit the screen
    original_h, original_w, _ = frames_data[0]['image'].shape
    num_frames = len(frames_data)
    grid_cols = int(math.ceil(math.sqrt(num_frames)))
    grid_rows = int(math.ceil(num_frames / grid_cols))
    frame_h, frame_w = original_h, original_w

    if (grid_cols * original_w > MAX_WINDOW_WIDTH) or (grid_rows * original_h > MAX_WINDOW_HEIGHT):
        scale = min(MAX_WINDOW_WIDTH / (grid_cols * original_w), MAX_WINDOW_HEIGHT / (grid_rows * original_h))
        frame_w, frame_h = int(original_w * scale), int(original_h * scale)
        for data in frames_data:
            data['image'] = cv2.resize(data['image'], (frame_w, frame_h), interpolation=cv2.INTER_AREA)

    # 5. Setup GUI and main loop
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse_event)
    print("\n--- Instructions ---")
    print("Click a frame to see its info in the console.")
    print("Click and drag a frame to reorder it.")
    print("Press ENTER to save the new order.")
    print("Press 'q' to quit.")

    while True:
        montage_h = grid_rows * frame_h
        montage_w = grid_cols * frame_w
        display_image = np.zeros((montage_h, montage_w, 3), dtype=np.uint8)

        for i, data in enumerate(frames_data):
            row, col = i // grid_cols, i % grid_cols
            y, x = row * frame_h, col * frame_w
            if is_dragging and i == drag_index:
                cv2.rectangle(display_image, (x, y), (x + frame_w, y + frame_h), (50, 50, 50), -1)
            else:
                display_image[y:y + frame_h, x:x + frame_w] = data['image']

        if is_dragging and drag_image_copy is not None:
            x_pos, y_pos = mouse_x - frame_w // 2, mouse_y - frame_h // 2
            x_start, y_start = max(x_pos, 0), max(y_pos, 0)
            x_end, y_end = min(x_pos + frame_w, montage_w), min(y_pos + frame_h, montage_h)
            img_x_start, img_y_start = x_start - x_pos, y_start - y_pos
            img_x_end, img_y_end = img_x_start + (x_end - x_start), img_y_start + (y_end - y_start)

            if x_end > x_start and y_end > y_start:
                 display_image[y_start:y_end, x_start:x_end] = drag_image_copy[img_y_start:img_y_end, img_x_start:img_x_end]

        cv2.imshow(window_name, display_image)
        key = cv2.waitKey(15) & 0xFF

        if key == ord('q'):
            print("\nExiting without saving.")
            break
        elif key == 13: # ENTER key
            final_ordered_paths = [data['path'] for data in frames_data]
            np.save(OUTPUT_PATHS_FILE, final_ordered_paths)
            print(f"\nSaved new order of {len(final_ordered_paths)} paths to '{OUTPUT_PATHS_FILE}'.")
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()