import numpy as np
import cv2

def gen_random_video(H, W, mean, var):
    """
    A generator that indefinitely generates frames of a video of size HxWx3
    with random pixels following a gaussian with given mean and variance.
    """
    while True:
        frame = np.random.normal(mean, var, size=(H, W, 3))
        # Clip values to be in the valid range for an image (0-255)
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        yield frame

# Use a fixed seed for reproducibility
np.random.seed(42)

# Define video and montage dimensions
VIDEO_H = 100
VIDEO_W = 200
MONTAGE_H = 1080
MONTAGE_W = 1920

TRANSITION_FRAMES = 400  # Number of frames for the zoom transition between levels
MAX_GRID_LEVEL = 7       # The highest grid level to reach (e.g., level 7 is a 15x15 grid)

# Create enough generators for all the videos we'll need
num_generators_needed = (2 * MAX_GRID_LEVEL + 1)**2
generators = [gen_random_video(VIDEO_H, VIDEO_W, np.random.rand() * 100 + 50, np.random.rand() * 20 + 5) for _ in range(num_generators_needed)]

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
        
        current_cell_w = int(VIDEO_W * render_zoom)
        current_cell_h = int(VIDEO_H * render_zoom)
        
        if current_cell_w <= 0 or current_cell_h <= 0:
            continue

        resized_frame = cv2.resize(video_frame, (current_cell_w, current_cell_h))

        paste_x = int(start_x_offset + (pos[0] + grid_level) * current_cell_w)
        paste_y = int(start_y_offset + (pos[1] + grid_level) * current_cell_h)

        paste_x_start = max(0, paste_x)
        paste_y_start = max(0, paste_y)
        paste_x_end = min(MONTAGE_W, paste_x + current_cell_w)
        paste_y_end = min(MONTAGE_H, paste_y + current_cell_h)

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

cv2.destroyAllWindows()
print("Video display finished.")