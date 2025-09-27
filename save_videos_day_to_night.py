import cv2
import numpy as np
import os

# --- Configuration ---
ORDERED_PATHS_FILE = 'ordered_frame_paths.npy'
ALL_PATHS_FILE = 'fred_frame_paths.npy'

OUTPUT_RGB_VIDEO = 'ordered_rgb_video_with_boxes.mp4'
OUTPUT_EVENTS_VIDEO = 'ordered_events_video_with_boxes.mp4'
FPS = 30.0 # Frames per second for the output videos

# --- HELPER FUNCTIONS FOR BOUNDING BOXES (from our first script) ---
to_timestamp = lambda idx: str(float("{:.6f}".format((idx+1)*0.033333)))

def read_coordinates(file_path):
    """
    Reads drone coordinates from a file with lines like:
    timestamp: x1, y1, x2, y2, idx, class_name
    Returns a dict: {timestamp: [[x1, y1, x2, y2, idx, class_name], ...]}
    """
    coordinates = {}
    if not os.path.exists(file_path):
        return coordinates # Return empty dict if file doesn't exist
    with open(file_path, "r") as f:
        for line in f:
            try:
                timestamp, rest = line.strip().split(": ")
                parts = rest.split(", ")
                x1, y1, x2, y2, idx = map(float, parts[:5])
                class_name = parts[5]

                if timestamp not in coordinates:
                    coordinates[timestamp] = []
                coordinates[timestamp].append([x1, y1, x2, y2, int(idx), class_name])
            except ValueError:
                # Handle cases where a line might be malformed
                continue
    return coordinates


def get_id_from_path(path):
    """Extracts the video ID from a filename like '.../Video_95_...jpg'."""
    try:
        filename = os.path.basename(path)
        return filename.split('_')[1]
    except IndexError:
        print(f"Warning: Could not extract video ID from path: {path}")
        return None


def main():
    """Main function to generate the ordered videos with bounding boxes."""
    # 1. Load the necessary data files
    print("Loading data files...")
    try:
        ordered_paths = np.load(ORDERED_PATHS_FILE, allow_pickle=True)
        all_paths_data = np.load(ALL_PATHS_FILE, allow_pickle=True).item()
    except FileNotFoundError as e:
        print(f"Error: Could not load a required file: {e.filename}")
        return

    # 2. Determine the unique, ordered sequence of video IDs
    ordered_ids = []
    seen_ids = set()
    for path in ordered_paths:
        video_id = get_id_from_path(path)
        if video_id and video_id not in seen_ids:
            ordered_ids.append(video_id)
            seen_ids.add(video_id)
    
    if not ordered_ids:
        print("Could not determine a valid order of video IDs. Exiting.")
        return
        
    print(f"Found {len(ordered_ids)} videos to process in the specified order.")

    # 3. Get video properties
    try:
        sample_path = ordered_paths[0]
        sample_frame = cv2.imread(sample_path)
        if sample_frame is None: raise IOError
        h, w, _ = sample_frame.shape
    except (IOError, IndexError):
        print("Error: Could not read a sample frame to determine video dimensions.")
        return

    # 4. Initialize video writers and the coordinate cache
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    rgb_writer = cv2.VideoWriter(OUTPUT_RGB_VIDEO, fourcc, FPS, (w, h))
    events_writer = cv2.VideoWriter(OUTPUT_EVENTS_VIDEO, fourcc, FPS, (w, h))
    writers = {'RGB': rgb_writer, 'Event': events_writer}
    output_files = {'RGB': OUTPUT_RGB_VIDEO, 'Event': OUTPUT_EVENTS_VIDEO}
    coodinate_cache = {}

    # 5. Process and write frames for both modalities
    for modality in ['RGB', 'Event']:
        print(f"\n--- Creating video for '{modality}' modality ---")
        writer = writers[modality]
        
        # Iterate through the videos in your specified order
        for video_id in ordered_ids + ordered_ids[-1:]*8:
            if int(video_id) in [91, 92, 93]:
                continue
            print(f"  Processing Video ID: {video_id}...")
            
            video_info = all_paths_data.get(video_id, {})
            frame_paths_list = video_info.get(modality, [])
            cut_index = int(video_info.get('cut_index', 0))
            
            if not frame_paths_list:
                print(f"    Warning: No '{modality}' frames found for Video ID {video_id}. Skipping.")
                continue

            # Modified loop to get frame_index for timestamp calculation
            for frame_index, frame_path in enumerate(frame_paths_list):
                # This keeps your logic of processing every 80th frame
                if frame_index % 80 != 0:
                    continue

                frame = cv2.imread(frame_path)
                if frame is not None:
                    # --- ADDING BOUNDING BOXES ---
                    # 1. Find and load coordinate data using cache
                    coordinate_file = frame_path.split(modality)[0] + 'coordinates.txt'
                    if coordinate_file not in coodinate_cache:
                        coodinate_cache[coordinate_file] = read_coordinates(coordinate_file)
                    coordinates = coodinate_cache[coordinate_file]

                    # 2. Calculate timestamp and check for coordinates
                    timestamp = to_timestamp(frame_index + cut_index)
                    if timestamp in coordinates:
                        for (x1, y1, x2, y2, obj_id, cls) in coordinates[timestamp]:
                            # 3. Draw rectangle and labels on the frame
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                            cv2.putText(frame, str(int(obj_id)), (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                            cv2.putText(frame, str(cls), (int(x1)+30, int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                    # --- END OF BOUNDING BOX LOGIC ---

                    # Ensure frame dimensions match the writer
                    if frame.shape[1] != w or frame.shape[0] != h:
                        frame = cv2.resize(frame, (w, h))
                    writer.write(frame)
        
        writer.release()
        print(f"--- Successfully saved '{output_files[modality]}' ---")

    print("\nAll videos have been created successfully! ✨")


if __name__ == '__main__':
    main()