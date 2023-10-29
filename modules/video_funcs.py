#########1#########2#########3#########4#########5#########6#########7#########
import os, errno, inspect 
import cv2, ffmpeg
from pathlib import Path
from PIL import Image

def get_default_args(func):
    """Getting default values of function arguments

    Args:
        func (func): function

    Returns:
        (dict): a dictionary {arg: value}
    """
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def combine_videochunks(chunks, sound_file, out_file, remove_chunks=False):
    """Combining video chunks and adding audio to the result

    Args:
        chunks (list): sorted list of video files to combine
        sound_file (str or PosixPath): video/audio file 
        out_file (str or PosixPath): output video file
        remove_chunks (bool, optional): chunks removal. Defaults to False.
    """
    ffmpeg_list = []
    for chunk in chunks:
        ffmpeg_list.append(ffmpeg.input(str(chunk)))
    if Path(out_file).is_file():
        out_file.unlink()
    
    a = ffmpeg.probe(str(sound_file), select_streams='a')
    if a['streams']:
        audio = ffmpeg.input(str(sound_file)).audio
        ffmpeg.concat(*ffmpeg_list).output(audio, str(out_file)).run()
    else:
        ffmpeg.concat(*ffmpeg_list).output(str(out_file)).run()
    
    if remove_chunks:
        for chunk in chunks:
            Path(chunk).unlink()
            

def check_video(fname):
    """ Convert input video file to appropriate format via ffmpeg

    Args:
        fname (str or PosixPath): video file

    Returns:
        PosixPath: name of the output file written to drive
    """
    fname = Path(fname)
    codec = ffmpeg.probe(fname)['streams'][0]['codec_name']
    out_fname = Path(str(fname.with_suffix('')) + '_tmp.mp4')
    if out_fname.is_file():
        out_fname.unlink()
    video = ffmpeg.input(str(fname))
    
    # av1 does not currently work in cv2
    if codec == 'av1':
        ffmpeg.output(video, str(out_fname)).run()
    else:
        ffmpeg.output(
            video, str(out_fname), acodec='copy', vcodec='copy').run()
    return out_fname


def write_video(frames, fname, fps, size: tuple):
    """Writing a video file from frames

    Args:
        frames (numpy array): frames (BGR colors)
        fname (str or PosixPath): output video file
        fps (float): frames per second
        size (tuple): output size [width, height]
    """
    video_out = cv2.VideoWriter(
        str(fname), cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    for frame in frames:
        video_out.write(frame)
    video_out.release()

    
def video2frames(video_file, out_dir):
    """Convert a video file to frame images saved in out_dir"""
    if not Path(video_file).is_file():
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), Path(video_file))
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    
    video = cv2.VideoCapture(str(video_file))
    count = 0
    while (video.isOpened()):
        ret, frame = video.read()
        if (ret != True):
            break
        filename = Path(out_dir)/str('frame%d.jpg' % count)
        count += 1
        cv2.imwrite(str(filename), frame)
            
    video.release()
    

def video2gif(video_file, gif_file, start, end, resize_factor=0.5):
    """Convert a video into a gif image from the start frame 
       to the end frame inclusive.
       Frame indexing is zero-based.
       Resize_factor resizes original size (e.g., 0.5 is for half-size)    
    """
    if not Path(video_file).is_file():
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), Path(video_file))
    
    video = cv2.VideoCapture(str(video_file))
    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (int(w*resize_factor), int(h*resize_factor))
    frame_duration = 1000 / video.get(cv2.CAP_PROP_FPS)
    video.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    while (video.get(cv2.CAP_PROP_POS_FRAMES) <= end):
        ret, frame = video.read()
        if (ret != True):
            break
        frames.append(Image.fromarray(frame[:,:,::-1]).resize(size))
    video.release()
        
    frames[0].save(
    gif_file,
    save_all=True,
    append_images=frames[1:], 
    optimize=True,
    duration=frame_duration,
    loop=0
    )    