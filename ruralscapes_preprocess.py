import os
import classmap
import imageio
import numpy as np
import torch


PATH_TO_ORIGINAL_LABELS = 'path_to_original_png_manual_labels'
OUTPUT_PATH = 'path_to_working_dir'

CLASS_COLORS_RGB = ((0, 255, 0), (0, 127, 0), (255, 255, 0), (255, 127, 0),
                    (255, 255, 255), (255, 0, 255), (127, 127, 127), (0, 0, 255),
                    (0, 255, 255), (127, 127, 63), (255, 0, 0), (127, 127, 0))


videos = sorted(os.listdir(PATH_TO_ORIGINAL_LABELS))

for video in videos:
    labels = sorted(os.listdir(os.path.join(PATH_TO_ORIGINAL_LABELS, video)))

    if not os.path.exists(os.path.join(OUTPUT_PATH, 'labels_2k', 'all', video)):
        os.makedirs(os.path.join(OUTPUT_PATH, 'labels_2k', 'all', video))
        os.makedirs(os.path.join(OUTPUT_PATH, 'labels_2k', 'train_even', video))
        os.makedirs(os.path.join(OUTPUT_PATH, 'labels_2k', 'train_odd', video))

    for idx, label in enumerate(labels):
        img = torch.tensor(imageio.imread(os.path.join(PATH_TO_ORIGINAL_LABELS, video, label)))
        mp = classmap.from_colors(img, CLASS_COLORS_RGB).numpy()[::2, ::2]

        name = label.replace('segfull_', '').replace('.png', '.npz')
        np.savez_compressed(os.path.join(OUTPUT_PATH, 'labels_2k', 'all', video, name), map=mp, votes=mp)
        if idx % 2:
            np.savez_compressed(os.path.join(OUTPUT_PATH, 'labels_2k', 'train_odd', video, name), map=mp, votes=mp)
        if not idx % 2:
            np.savez_compressed(os.path.join(OUTPUT_PATH, 'labels_2k', 'train_even', video, name), map=mp, votes=mp)
