import numpy as np
import cv2
import json
import os
import time

to_timestamp = lambda idx: str(float("{:.6f}".format((idx+1)*0.033333)))

def read_coordinates(file_path):
    """
    Reads coordinates from a file with lines like:
    timestamp: x1, y1, x2, y2, idx, class_name
    Returns a dict: {timestamp: [[x1, y1, x2, y2, idx, class_name], ...]}
    """
    coordinates = {}
    with open(file_path, "r") as f:
        for line in f:
            timestamp, rest = line.strip().split(": ")
            parts = rest.split(", ")
            x1, y1, x2, y2, idx = map(float, parts[:5])
            class_name = parts[5]

            if timestamp not in coordinates:
                coordinates[timestamp] = []
            coordinates[timestamp].append([x1, y1, x2, y2, int(idx), class_name])

    #unique_ids = {coord[4] for ts in coordinates for coord in coordinates[ts]}
    #print(f"Number of unique ids: {len(unique_ids)}")
    return coordinates

coodinate_cache = {}

def frame_generator(cur_frame_paths):
    """
    A generator that yields frames from a list of image file paths.
    """
    modality = np.random.choice(['Event', 'RGB'], p=[0.5, 0.5])
    while True:
        for n, frame_path in enumerate(cur_frame_paths[modality]):
            # read frame
            frame = cv2.imread(frame_path)

            # read coordinates
            coordinate_file = frame_path.split(modality)[0] + 'coordinates.txt'
            if coordinate_file not in coodinate_cache:
                coodinate_cache[coordinate_file] = read_coordinates(coordinate_file)
            coordinates = coodinate_cache[coordinate_file]
            timestamp = to_timestamp(n + int(cur_frame_paths['cut_index']) + 1)
            if timestamp in coordinates:
                for (x1, y1, x2, y2, obj_id, cls) in coordinates[timestamp]:
                    # Adjust for crop
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # add idx to the rectangle
                    cv2.putText(frame, str(int(obj_id)), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                    cv2.putText(frame, str(cls), (x1+20, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

            # crop frame
            frame = frame[100:-100, 100:-100][:,:,::-1]
            
            # halve size
            #frame = cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))
            yield frame

frame_paths = np.load('fred_frame_paths.npy', allow_pickle=True).item()

# Define video and montage dimensions
VIDEO_H = 100
VIDEO_W = 200
MONTAGE_H = 1080
MONTAGE_W = 1920

TRANSITION_FRAMES = 100  # Number of frames for the zoom transition between levels
MAX_GRID_LEVEL = 7       # The highest grid level to reach (e.g., level 7 is a 15x15 grid)
save = True

if save:
    # insert timestamp in video name
    out_video_path = f'video_wall_fred_{int(time.time())}.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(out_video_path, fourcc, 30.0, (MONTAGE_W, MONTAGE_H))


# Create enough generators for all the videos we'll need
num_generators_needed = (2 * MAX_GRID_LEVEL + 1)**2
generators = [frame_generator(x) for x in frame_paths.values()]

# shuffle generators
np.random.shuffle(generators)

# --- Video Montage Logic ---

# Keep track of active video cells
active_cells = {
    (0, 0): generators[0]  # Start with the central video
}
gen_index = 1

# Animation state variables
grid_level = 0
transition_progress = 0

print("Generating video montage...")
print("Press 'q' or close the window to exit.")

while True:
    # --- State and Zoom Calculation ---
    
    # At the beginning of each transition, increment the grid level and add new videos
    if transition_progress == 0:
        grid_level += 1
        if grid_level > MAX_GRID_LEVEL:
            break  # End of the animation

        min_coord, max_coord = -grid_level, grid_level
        
        # Add the new videos to the perimeter of the expanding grid
        for x in range(min_coord, max_coord + 1):
            if (x, min_coord) not in active_cells and gen_index < len(generators):
                active_cells[(x, min_coord)] = generators[gen_index]; gen_index += 1
            if (x, max_coord) not in active_cells and gen_index < len(generators):
                active_cells[(x, max_coord)] = generators[gen_index]; gen_index += 1
        
        for y in range(min_coord + 1, max_coord):
            if (min_coord, y) not in active_cells and gen_index < len(generators):
                active_cells[(min_coord, y)] = generators[gen_index]; gen_index += 1
            if (max_coord, y) not in active_cells and gen_index < len(generators):
                active_cells[(max_coord, y)] = generators[gen_index]; gen_index += 1

    # The grid size (in videos) we are transitioning FROM
    start_grid_dim = 2 * (grid_level - 1) + 1
    # The grid size we are transitioning TO
    end_grid_dim = 2 * grid_level + 1

    # Calculate the zoom needed for the start and end grids to fill the screen.
    start_zoom = max(MONTAGE_W / (start_grid_dim * VIDEO_W), MONTAGE_H / (start_grid_dim * VIDEO_H))
    end_zoom = max(MONTAGE_W / (end_grid_dim * VIDEO_W), MONTAGE_H / (end_grid_dim * VIDEO_H))

    # 't' is our progress (0.0 to 1.0) through the current transition
    t = transition_progress / TRANSITION_FRAMES
    
    ## FIX: The following line created the non-linear "acceleration" effect.
    ## It has been removed to make the zoom change linear.
    # t = t * t * (3.0 - 2.0 * t)
    
    # Interpolate the zoom factor for the current frame
    render_zoom = start_zoom + t * (end_zoom - start_zoom)

    # --- Rendering the Montage on a Fixed Canvas ---
    montage_frame = np.zeros((MONTAGE_H, MONTAGE_W, 3), dtype=np.uint8)
    
    # Calculate the total grid size in pixels to correctly center it
    grid_w_pixels = end_grid_dim * VIDEO_W * render_zoom
    grid_h_pixels = end_grid_dim * VIDEO_H * render_zoom
    
    start_x_offset = (MONTAGE_W - grid_w_pixels) / 2
    start_y_offset = (MONTAGE_H - grid_h_pixels) / 2

    for pos, gen in active_cells.items():
        video_frame = next(gen)
        
       # 1. Mantieni le dimensioni esatte (float) per i calcoli di posizione
        cell_w_float = VIDEO_W * render_zoom
        cell_h_float = VIDEO_H * render_zoom

        # 1. Calcola le coordinate di inizio e fine della cella usando i float
        paste_x_start_float = start_x_offset + (pos[0] + grid_level) * cell_w_float
        paste_y_start_float = start_y_offset + (pos[1] + grid_level) * cell_h_float
        
        # La coordinata "end" è semplicemente la coordinata "start" della cella successiva
        paste_x_end_float = start_x_offset + (pos[0] + grid_level + 1) * cell_w_float
        paste_y_end_float = start_y_offset + (pos[1] + grid_level + 1) * cell_h_float

        # 2. Converti le coordinate in posizioni di pixel (interi)
        paste_x = int(paste_x_start_float)
        paste_y = int(paste_y_start_float)

        # 3. La dimensione del frame è la differenza tra le posizioni dei pixel di fine e inizio.
        #    Questo garantisce che non ci siano buchi!
        resized_cell_w = int(paste_x_end_float) - paste_x
        resized_cell_h = int(paste_y_end_float) - paste_y

        if resized_cell_w <= 0 or resized_cell_h <= 0:
            continue

        resized_frame = cv2.resize(video_frame, (resized_cell_w, resized_cell_h))

        # 3. Calcola la posizione usando i valori float e converti in int solo alla fine
        paste_x = int(start_x_offset + (pos[0] + grid_level) * cell_w_float)
        paste_y = int(start_y_offset + (pos[1] + grid_level) * cell_h_float)

        # Il resto della logica per incollare il frame rimane invariato
        paste_x_start = max(0, paste_x)
        paste_y_start = max(0, paste_y)
        paste_x_end = min(MONTAGE_W, paste_x + resized_cell_w)
        paste_y_end = min(MONTAGE_H, paste_y + resized_cell_h)

        slice_x_start = paste_x_start - paste_x
        slice_y_start = paste_y_start - paste_y
        slice_x_end = slice_x_start + (paste_x_end - paste_x_start)
        slice_y_end = slice_y_start + (paste_y_end - paste_y_start)

        if slice_x_end > slice_x_start and slice_y_end > slice_y_start:
            montage_frame[paste_y_start:paste_y_end, paste_x_start:paste_x_end] = \
                resized_frame[slice_y_start:slice_y_end, slice_x_start:slice_x_end]
    
    cv2.imshow('Video Montage', cv2.cvtColor(montage_frame, cv2.COLOR_RGB2BGR))
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    transition_progress = (transition_progress + 1) % TRANSITION_FRAMES

    if save:
        out_video.write(cv2.cvtColor(montage_frame, cv2.COLOR_RGB2BGR))

if save:
    out_video.release()

cv2.destroyAllWindows()
print("Video display finished.")