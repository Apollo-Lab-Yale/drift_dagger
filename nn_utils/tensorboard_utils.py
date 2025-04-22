import numpy as np
from moviepy.editor import ImageSequenceClip

def save_video_from_tensor(tensor, output_path, fps=30):
    video_array = tensor.numpy()
    video_array = np.transpose(video_array, (0, 1, 3, 4, 2))
    video_array = video_array.squeeze(0)
    frames = [video_array[i] for i in range(video_array.shape[0])]
    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(output_path, codec='libx264')