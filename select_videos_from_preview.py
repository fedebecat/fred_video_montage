import cv2
import numpy as np
import os
import math

# --- Configuration ---
FRAME_INDEX_TO_LOAD = 500  # 50th frame is at index 49
BORDER_THICKNESS = 5
BORDER_COLOR_BGR = (0, 255, 0)  # Green
OUTPUT_FILENAME = 'selected_video_ids.npy'

# --- NEW: Set the maximum dimensions for the final window ---
MAX_WINDOW_WIDTH = 1920
MAX_WINDOW_HEIGHT = 1080


# --- Global variables to store the application's state ---
frames_data = []
selected_ids = set()
base_montage = None
frame_h, frame_w = 0, 0
grid_cols = 0
window_name = 'Frame Montage Selector'

def on_mouse_click(event, x, y, flags, param):
    """Callback function to handle mouse clicks on the montage window."""
    global selected_ids, frames_data, grid_cols, frame_h, frame_w

    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if frame dimensions are valid to prevent division by zero
        if frame_w == 0 or frame_h == 0:
            return

        col = x // frame_w
        row = y // frame_h
        index = row * grid_cols + col

        if 0 <= index < len(frames_data):
            video_id = frames_data[index]['id']
            if video_id in selected_ids:
                selected_ids.remove(video_id)
                print(f"Deselected: {video_id}")
            else:
                selected_ids.add(video_id)
                print(f"Selected: {video_id}")

def main():
    """Main function to run the application."""
    global frames_data, base_montage, frame_h, frame_w, grid_cols

    # 1. Load the frame paths data
    try:
        all_paths = np.load('fred_frame_paths.npy', allow_pickle=True).item()
    except FileNotFoundError:
        print(f"Error: 'fred_frame_paths.npy' not found. Make sure it's in the same directory.")
        return

    # 2. Extract the specified frame from each RGB video source
    print(f"Loading frame {FRAME_INDEX_TO_LOAD + 1} from each RGB video...")
    for video_id, paths_dict in all_paths.items():
        if 'RGB' in paths_dict and len(paths_dict['RGB']) > FRAME_INDEX_TO_LOAD:
            frame_path = paths_dict['RGB'][FRAME_INDEX_TO_LOAD]
            if os.path.exists(frame_path):
                image = cv2.imread(frame_path)
                if image is not None:
                    frames_data.append({'id': video_id, 'image': image})

    if not frames_data:
        print("No valid frames were found to create a montage. Exiting.")
        return
    print(f"Successfully loaded {len(frames_data)} frames.")

    # 3. Calculate montage grid and initial dimensions
    num_frames = len(frames_data)
    original_h, original_w, _ = frames_data[0]['image'].shape
    grid_cols = int(math.ceil(math.sqrt(num_frames)))
    grid_rows = int(math.ceil(num_frames / grid_cols))
    
    # Set the global frame dimensions, which may be updated
    frame_h, frame_w = original_h, original_w

    # --- NEW: Logic to resize frames if the montage is too large ---
    total_w = grid_cols * original_w
    total_h = grid_rows * original_h

    if total_w > MAX_WINDOW_WIDTH or total_h > MAX_WINDOW_HEIGHT:
        print("Montage is too large for the screen. Resizing frames...")
        # Calculate the scaling factor needed to fit the window
        scale_w = MAX_WINDOW_WIDTH / total_w
        scale_h = MAX_WINDOW_HEIGHT / total_h
        scale = min(scale_w, scale_h) # Use the smaller scale to ensure it fits both ways

        # Calculate the new, smaller dimensions for each frame
        frame_w = int(original_w * scale)
        frame_h = int(original_h * scale)
        print(f"New frame size: {frame_w}x{frame_h}")

        # Resize all loaded images in place
        for data in frames_data:
            data['image'] = cv2.resize(data['image'], (frame_w, frame_h), interpolation=cv2.INTER_AREA)

    # 4. Create the base montage using the (potentially new) frame dimensions
    montage_h = grid_rows * frame_h
    montage_w = grid_cols * frame_w
    base_montage = np.zeros((montage_h, montage_w, 3), dtype=np.uint8)

    print("Creating montage image...")
    for i, data in enumerate(frames_data):
        row = i // grid_cols
        col = i % grid_cols
        y_start, x_start = row * frame_h, col * frame_w
        base_montage[y_start:y_start+frame_h, x_start:x_start+frame_w] = data['image']
        data['rect'] = (x_start, y_start, x_start + frame_w, y_start + frame_h)

    # 5. Setup GUI window and the main interaction loop
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse_click)

    print("\n--- Instructions ---")
    print("Click on a frame to select/deselect it.")
    print("Press ENTER to save the IDs of selected frames.")
    print("Press 'q' to quit without saving.")

    while True:
        display_image = base_montage.copy()
        for data in frames_data:
            if data['id'] in selected_ids:
                x1, y1, x2, y2 = data['rect']
                cv2.rectangle(display_image, (x1, y1), (x2, y2), BORDER_COLOR_BGR, BORDER_THICKNESS)
        
        cv2.imshow(window_name, display_image)
        key = cv2.waitKey(20) & 0xFF

        if key == ord('q'):
            print("\nExiting without saving.")
            break
        elif key == 13:
            selected_list = list(selected_ids)
            np.save(OUTPUT_FILENAME, selected_list)
            print(f"\nSaved {len(selected_list)} selected video IDs to '{OUTPUT_FILENAME}'.")
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()