""" Motion synchronicity scoring in a video file
   
    usage: skeletons.py [-h] [-b BATCH_SIZE] [--score | --no-score] in_videofile out_videofile
 
    positional arguments:
      in_videofile          Input video file
      out_videofile         Output video file
    
    options:
      -h, --help            show this help message and exit
      -b BATCH_SIZE, --batch_size BATCH_SIZE
                            Batch size (default: 4)
      --score, --no-score   Get synchronicity score
"""
# Adding local modules to the search path
import sys
from pathlib import Path
lib = Path(__file__).parent/'modules'
sys.path.insert(0, str(lib))

# python env modules
import numpy as np
import argparse, cv2
from torch.utils.data import DataLoader
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from tqdm import tqdm

# local modules
from skeleton_studio import FrameDataset, DancingSkeletons 
from video_funcs import (
    get_default_args, check_video, combine_videochunks, write_video)


def main(video_file, out_file, batch_size, get_score):
    # Initialization
    NFRAMES_WRITE = 1000  # No. of frames to write to drive
    video_file = check_video(Path(video_file))
    out_file = Path(out_file)
    video = cv2.VideoCapture(str(video_file))
    video_w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = video.get(cv2.CAP_PROP_FPS)
    
    # Data loaders and model
    dataset = FrameDataset(video, batch_size=batch_size)
    dataloader = DataLoader(dataset, batch_size=None)
    skeletons = DancingSkeletons(keypointrcnn_resnet50_fpn(weights='DEFAULT'))

    # video chunks to write to drive
    chunk_n = 0
    frames_out, count = [], 0

    # Batch inference and writing chunks to drive
    simscores = np.array([])  # all similarity scores
    for eof, batch in tqdm(dataloader):
        if len(batch) > 0:
            preds = skeletons.inference(batch)
            f_with_skeletons, simscores = skeletons.draw_batch(
                batch, preds, simscores, get_score=get_score)
            count += batch_size
            frames_out.extend(f_with_skeletons)
    
        # if chunk is complete or end of file
        if (count%NFRAMES_WRITE==0 or eof) and len(frames_out)>0:
            print(f' Writing chunk {chunk_n}')
            chunk_name = out_file.parent/('tmp_chunk' + str(chunk_n) + '.mp4')
            write_video(frames_out, chunk_name, video_fps, (video_w, video_h))
            chunk_n += 1
            frames_out = []
    
    video.release()

    # Combining video chunks and audio from the original video file
    chunks = sorted(list(out_file.parent.glob('tmp_chunk*.mp4')))
    if get_score:
        # Final score video chunk
        final_chunk = out_file.parent/('tmp_fchunk.mp4')
        final_frame = np.zeros(shape=(video_h, video_w, 3), dtype=np.uint8)
        text = f'Total score: {int(np.nanmean(simscores)):02d}'
        
        # Text layout from function default values 
        default_scalers = get_default_args(DancingSkeletons.print_score)
        text_color = default_scalers['text_color'] 
        font_scale = default_scalers['font_scale'] 
        thickness_scale = default_scalers['thickness_scale']
        x_pos = default_scalers['x_pos'] 
        y_pos_scale = default_scalers['y_pos_scale'] 
        
        fontscale = min(video_h, video_w) * font_scale
        thickness = int((min(video_h, video_w) * thickness_scale))
        y_pos = y_pos_scale * thickness
        
        # Making the final chunk
        cv2.putText(
            final_frame, text, (x_pos, y_pos), 
            fontFace=cv2.FONT_ITALIC, fontScale=fontscale, 
            color=text_color, thickness=thickness
        )
        fin_frames = [final_frame for _ in range(int(5 * video_fps))]
        write_video(fin_frames, final_chunk, video_fps, (video_w, video_h))
        chunks.append(final_chunk)
    
    # Combining video chunks and audio from the original video file
    combine_videochunks(chunks, video_file, out_file, remove_chunks=True)
    video_file.unlink()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Motion synchronicity scoring')
    parser.add_argument('in_videofile', type=str, help='Input video file')
    parser.add_argument('out_videofile', type=str, help='Output video file')
    parser.add_argument(
        '-b', '--batch_size', type=int, default=4, help='Batch size (default: 4)'
    )
    parser.add_argument(
        '--score', default=True, action=argparse.BooleanOptionalAction,
        help='Get synchronicity score')
    args = parser.parse_args()

    main(args.in_videofile, args.out_videofile, args.batch_size, args.score)