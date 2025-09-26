from glob import glob
from natsort import natsorted
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

dataset_path = '/media/becattini/SSD4TB/datasets/FRED/'

splits = ['train', 'test']

video_folders = []
for s in splits:
    video_folders_s = glob(f'{dataset_path}/{s}/*/')
    video_folders.extend(video_folders_s)

print(len(video_folders))

get_event_frame_paths = lambda x: natsorted(glob(f'{x}/Event/Frames/*.png'))
get_rgb_frame_paths = lambda x: natsorted(glob(f'{x}/RGB/*.jpg'))
get_frame_paths = lambda x, y: get_event_frame_paths(x) if y == 'Event' else get_rgb_frame_paths(x)

get_file_weight_bytes = lambda x: os.path.getsize(x)


# -----------------------------------------

frame_path_dict = {}

for cur_video_folder in video_folders:
    print(f'Inspecting video folder: {cur_video_folder}')
    print(f'cur video folder {cur_video_folder}')

    video_id = cur_video_folder.split('/')[-2]
    print(f'video id: {video_id}')

    # if video_id != '17':
    #     continue

    event_paths = get_frame_paths(cur_video_folder, 'Event')
    rgb_paths = get_frame_paths(cur_video_folder, 'RGB')

    # sanity check
    assert len(event_paths) == len(rgb_paths)
    
    # find file weight in bytes for all files in folder
    event_weights = [get_file_weight_bytes(x) for x in event_paths]

    # remove initial burst of frames with high weight
    median_event_weight = int(np.median(event_weights))
    print(f'Median event weight: {median_event_weight}')

    # plt.plot(event_weights,'.', markersize=1)
    # plt.axhline(y=median_event_weight, color='r', linestyle='--')
    # plt.show()

    cut_index = np.min(np.where(np.array(event_weights) < median_event_weight)[0])
    print(f'Cut index: {cut_index}')

    frame_path_dict[video_id] = {
        'Event': event_paths[cut_index:],
        'RGB': rgb_paths[cut_index:],
        'cut_index': str(cut_index)
    }

    # # show video
    # for i in range(cut_index, len(event_paths)):
    #     event_frame = cv2.imread(event_paths[i])
    #     rgb_frame = cv2.imread(rgb_paths[i])

    #     event_weight = 1
    #     rgb_weight = 1 - event_weight
    #     combined_frame = cv2.addWeighted(event_frame, event_weight, rgb_frame, rgb_weight, 0)

    #     # remove salt and pepper noise from event frame for better visualization
    #     event_frame_gray = cv2.cvtColor(event_frame, cv2.COLOR_BGR2GRAY)
    #     _, event_frame_binary = cv2.threshold(event_frame_gray, 30, 255, cv2.THRESH_BINARY)
    #     event_frame_binary = cv2.medianBlur(event_frame_binary, 25)
    #     event_frame_clean = cv2.bitwise_and(event_frame, event_frame, mask=event_frame_binary)
    #     combined_frame = cv2.addWeighted(event_frame_clean, event_weight, rgb_frame, rgb_weight, 0)
        
    #     cv2.imshow('combined', combined_frame)
    #     k = cv2.waitKey(1)
    #     if k == ord('q'):
    #         break

# saved dictionary as json
# import json
# with open('fred_frame_paths.json', 'w') as f:
#     json.dump(frame_path_dict, f)
np.save('fred_frame_paths.npy', frame_path_dict)