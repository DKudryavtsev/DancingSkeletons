#########1#########2#########3#########4#########5#########6#########7#########
import os, errno, cv2, torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from torchvision import transforms
from torch.utils.data import IterableDataset
from pathlib import Path


class DancingSkeletons():
    """Batch inference and drawing multiple instances on frames;
       similarity score calculation
    """
    def __init__(self, model):
        """Class instance initialization

        Args:
            model (func): torchvision keypointrcnn_resnet50_fpn model
                          or an analog  with the same output format
        """
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.eval().to(self.device)
        
        self.keypoints = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 
            'right_knee', 'left_ankle', 'right_ankle']
        
        # Keypoints for similarity score calculation
        self.keypoints_scored_ids = [
            self.keypoints.index('nose'), 
            self.keypoints.index('left_elbow'), 
            self.keypoints.index('right_elbow'), 
            self.keypoints.index('left_wrist'), 
            self.keypoints.index('right_wrist'),
            self.keypoints.index('left_knee'), 
            self.keypoints.index('right_knee'), 
            self.keypoints.index('left_ankle'), 
            self.keypoints.index('right_ankle')]
        
        self.joints = [
            [self.keypoints.index('right_eye'), 
             self.keypoints.index('nose')],
            [self.keypoints.index('right_eye'), 
             self.keypoints.index('right_ear')],
            [self.keypoints.index('left_eye'), 
             self.keypoints.index('nose')],
            [self.keypoints.index('left_eye'), 
             self.keypoints.index('left_ear')],
            [self.keypoints.index('right_shoulder'), 
             self.keypoints.index('right_elbow')],
            [self.keypoints.index('right_elbow'), 
             self.keypoints.index('right_wrist')],
            [self.keypoints.index('left_shoulder'), 
             self.keypoints.index('left_elbow')],
            [self.keypoints.index('left_elbow'), 
             self.keypoints.index('left_wrist')],
            [self.keypoints.index('right_hip'), 
             self.keypoints.index('right_knee')],
            [self.keypoints.index('right_knee'), 
             self.keypoints.index('right_ankle')],
            [self.keypoints.index('left_hip'), 
             self.keypoints.index('left_knee')],
            [self.keypoints.index('left_knee'), 
             self.keypoints.index('left_ankle')],
            [self.keypoints.index('right_shoulder'), 
             self.keypoints.index('left_shoulder')],
            [self.keypoints.index('right_hip'), 
             self.keypoints.index('left_hip')],
            [self.keypoints.index('right_shoulder'), 
             self.keypoints.index('right_hip')],
            [self.keypoints.index('left_shoulder'), 
             self.keypoints.index('left_hip')],
        ]

    def inference(self, tensors_list):
        """Inference

        Args:
            tensors_list (torch Tensor): input frames

        Returns:
            numpy.ndarrays: boxes, labels, scores, keypoints, keypoints_scores
            as for PyTorch keypointrcnn_resnet50_fpn model
        """
        with torch.no_grad():
            predictions = self.model(tensors_list)
        return predictions
    
    def lstsq_affine_transform(self, kp_ref, kp_test, ref_confs, test_confs):
        """Affine transformation of a 'test' instance to the 'reference'
           Best match is calculated by weighted least squares

        Args:
            kp_ref (np.ndarray): a [17, 2] array with keypoint coordinates
                                 for the reference instance
            kp_test (np.ndarray): same for the test instance
            ref_confs (np.ndarray): keypoint confidence scores (reference)
            test_confs (np.ndarray): same for the test instance

        Returns:
            np.ndarray: transformed keypoints for the test instance
        """
        # X*A + b = Y  ->  X * A = Y
        pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
        unpad = lambda x: x[:, :-1]
        
        # Negative keypoint confidences to zero
        ref_confs[ref_confs < 0 ] = 0
        test_confs[test_confs < 0 ] = 0
        
        # Best affine transformation from weighted least squares
        W = np.sqrt(np.diag(ref_confs * test_confs))
        X = W @ pad(kp_test) 
        Y = W @ pad(kp_ref)
        A, res, rank, s = np.linalg.lstsq(X, Y, rcond=None)
        A[np.abs(A) < 1e-10] = 0  # low values to zero
        
        transform = lambda x: unpad(pad(x) @ A)
        return transform(kp_test)
    
    def normalize_array(self, array, axis=0):
        """Centering along an axis and normalization as a total.
           Used in pose similarity estimation

        Args:
            array (np.ndarray): input array (e.g., keypoint coordinates)
            axis (int, optional): array axis along which the centering 
                                  is performed. Defaults to 0.

        Returns:
            np.ndarray: normalized array
        """
        array = array - array.mean(axis=axis)
        return array / np.linalg.norm(array)

    def calculate_simscore(
        self, confs, keypoints, keypoint_confs, score_bias=0.05, 
        score_scale=0.1, top_score=10
        ):
        """Pose similarity score for multiple instances in a frame.
           Based on weighted keypoint distances between the reference figure 
           and the others. The reference figure is the first instance
           (confs are sorted in the network output).
           Scoring for only selected keypoints: self.keypoints_scored_ids
           All input arrays must be true detections.

        Args:
            confs (np.ndarray): confidence scores for the instances
            keypoints ((np.ndarray): keipoints [n_inst, 17, 2], 
            keypoint_confs (np.ndarray): keipoint conf. scores [n_inst, 17]
            
            Distance-to-score calculation:
            score_bias (float, optional): distances histogram shift, ~Q1;
                                          (add for better score). 
                                          Defaults to 0.05.
            score_scale (float, optional): distances range, ~histogram width;
                                          (add for tighter range). 
                                          Defaults to 0.1.
            top_score (int, optional): Highest possible score. Defaults to 10.

        Returns:
            float: Pose similarity score (0 - top_score)
        """
        # No score for No. of persons == 0 or 1
        if len(confs) < 2:
            return np.NaN
        
        # First instance in confs is the reference pose
        pose_ref = self.normalize_array(keypoints[0])
        personal_scores = []
        
        # For each instance
        for person_id in range(1, len(confs)):
            # Affine transformation of the test to the reference
            pose_test = self.lstsq_affine_transform(
                keypoints[0], keypoints[person_id], 
                keypoint_confs[0], keypoint_confs[person_id])
            
            # Normalization
            pose_test = self.normalize_array(pose_test)

            # Scoring based on weighted distances between keypoints 
            mutual_confs = np.sqrt(
                keypoint_confs[0] * keypoint_confs[person_id])
            sum, denom = 0, 0
            for kp in self.keypoints_scored_ids:
                sum += (mutual_confs[kp] 
                        * np.linalg.norm(pose_test[kp]-pose_ref[kp]))
                denom += mutual_confs[kp]
            # Avoiding errors due to low conf. instances
            if denom!=0 and ~np.isnan(denom):
                personal_scores.append(sum / denom)
        
        frame_score = (
            (1-(np.mean(personal_scores)-score_bias)/score_scale) * top_score)
        return max(0, min(frame_score, top_score))

    def draw_frame(self, frame, all_keypoints, all_scores, keypoint_threshold):
        """Drawing keypoints and joints for instances in a frame

        Args:
            frame (np.ndarray): image
            all_keypoints (np.ndarray): keypoints [n_instances, 17, 2]
            all_scores (np.ndarray): keypoint conf. scores [n_instances, 17]
            keypoint_threshold (float): threshold for the scores

        Returns:
            np.ndarray: frame with skeletons drawn
        """
        # color sequence 
        n_instances = all_keypoints.shape[0]
        cmap = plt.get_cmap('jet')
        color_ids = (np.linspace(0, 255, n_instances + 2)
                     .astype(int).tolist()[1:-1])

        # for each instance from left to right
        color_order = 0  # color order in color_ids
        for person_id in all_keypoints[:,:,0].mean(axis=1).argsort():
            keypoints = all_keypoints[person_id, ...]
            scores = all_scores[person_id, ...]
            color = tuple(
                (np.asarray(cmap(color_ids[color_order])[:-1]) * 255))
            color_order += 1
            
            # for each keepoint
            for kp_id, score in enumerate(scores):
                if score > keypoint_threshold:
                    keypoint = tuple(map(int, keypoints[kp_id]))
                    cv2.circle(frame, keypoint, 5, color, -1)
                      
            # for each joint
            for joint in self.joints:
                kp1_id,  kp2_id = joint[0], joint[1]
                if (scores[kp1_id] > keypoint_threshold 
                    and  scores[kp2_id] > keypoint_threshold
                ):
                    keypoint1 = tuple(map(int, keypoints[kp1_id]))
                    keypoint2 = tuple(map(int, keypoints[kp2_id]))
                    cv2.line(frame, keypoint1, keypoint2,
                             color, thickness=2)
            
        return frame
    
    
    def print_score(
        self, frame, text, text_color=(68, 96, 255), font_scale=1.5e-3, 
        thickness_scale=5e-3, x_pos=10, y_pos_scale=13
        ):
        """Writing score information in a frame

        Args:
            frame (np.ndarray): frame (BGR)
            text (str): text
            text_color (tuple, optional): text color (BGR). 
                                          Defaults to (68, 96, 255), Neon coral
            
            Text font and position scalers for image size to font size calc.:                           
            font_scale (float, optional): Defaults to 1.5e-3.
            thickness_scale (float, optional): Defaults to 5e-3.
            x_pos (int, optional): text start position in px. Defaults to 10.
            y_pos_scale (int, optional): scale, not position! Defaults to 13.
        Returns:
            np.ndarray: frame with score inprinted (BGR)
        """
        fontscale = min(frame.shape[:2]) * font_scale
        thickness = int((min(frame.shape[:2]) * thickness_scale))
        y_pos = y_pos_scale * thickness
        cv2.putText(
            frame, text, (x_pos, y_pos), 
            fontFace=cv2.FONT_ITALIC, fontScale=fontscale, 
            color=text_color, thickness=thickness
        )
        
        return frame
    
    
    def draw_batch(self, frames, preds, simscores, conf_threshold=0.9,
                   keypoint_threshold=2, get_score=True, score_ma_range=10):
        """Draw skeletons and scores in a batch of frames

        Args:
            frames (np.ndarray): frames from dataloader, [B, C, H, W] (RGB)
            preds (list): neural networks inference output
            simscores (np.ndarray): similarity scores for previous frames
            conf_threshold (float, optional): instance detection threshold. 
                                              Defaults to 0.9.
            keypoint_threshold (int, optional): keypoint detection threshold.
                                                Defaults to 2.
            get_score (bool, optional): score calculation. Defaults to True.
            score_ma_range (int, optional): moving average range for score 
                                            printing. Defaults to 10.

        Returns:
            tuple: output frames (BGR) and renewed simscores
        """
        # Frames back to cv2-style BGR color
        frames = [np.ascontiguousarray(
            frame.cpu().numpy().transpose(1, 2, 0)[:,:,::-1] * 255, dtype=np.uint8)
            for frame in frames
        ]       
        out_frames = []

        # for each frame
        for frame_id, pred in enumerate(preds):  
            frame = frames[frame_id]
            boxes, labels, confs, all_keypoints, all_scores = (
                [x.cpu().numpy() for x in pred.values()])
            
            # delete instances with low confidence
            true_mask = confs > conf_threshold 
            confs, all_keypoints, all_scores = (
                confs[true_mask], all_keypoints[true_mask], all_scores[true_mask])
            n_instances = len(confs)

            if n_instances != 0:
                # delete neural network instance label
                all_keypoints = all_keypoints[:,:,:2]
                
                # Draw skeletons in a frame
                frame = self.draw_frame(
                    frame, all_keypoints, all_scores, keypoint_threshold)
                
                # Calculate and write similarity score for a frame
                if get_score:
                    simscore = self.calculate_simscore(
                        confs, all_keypoints, all_scores)
                    simscores = np.append(simscores, simscore)
                    
                    # moving average for score printing
                    ma_range = simscores[-score_ma_range:]
                    ma_range = ma_range[~np.isnan(ma_range)]
                    if ma_range.shape[0] != 0:
                        moving_aver = np.mean(ma_range)
                    else:
                        moving_aver = np.NaN
                    
                    # text to print
                    if n_instances == 1:
                        text = 'One person'
                    elif ~np.isnan(moving_aver):
                        #text = f'Similarity: {int(moving_aver)}%'
                        text = f'Similarity score: {int(moving_aver):02d}'
                    else:
                        text = ''
                    frame = self.print_score(frame, text)
                                    
            out_frames.append(frame)
        return out_frames, simscores


class FrameDataset(IterableDataset):
    """Torch iterable dataset from a video file 
    """
    def __init__(self, video_stream, batch_size):
        """Instance initialization

        Args:
            video_stream (cv2.VideoCapture): video stream
            batch_size (int): batch size
        """
        super(FrameDataset).__init__()
        self.video = video_stream
        self.batch_size = batch_size
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.length = self.video.get(cv2.CAP_PROP_FRAME_COUNT)
        self.n_batches = int(np.ceil(self.length / self.batch_size))
        print(f'Video will be processed in {self.n_batches} batches')
        self.EOF = False
    
    def process_frame(self, frame):
        # frames to RGB color
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        transforms_ = transforms.ToTensor()
        tensor = transforms_(frame).to(self.device)
        return tensor
        
    def batch_generator(self):
        EOF = False
        while not EOF:
            batch_out = []
            for _ in range(self.batch_size):
                ret, frame = (self.video).read()
                if ret:
                    batch_out.append(self.process_frame(frame))
                else:
                    EOF = True
            yield EOF, batch_out
                
    def __iter__(self):  
        return self.batch_generator()


class Skeleton():
    """Keypoints inference and drawing for a single image 
    """
    def __init__(self):
                
        self.keypoints = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 
            'right_knee', 'left_ankle', 'right_ankle']
        
        self.joints = [
            [self.keypoints.index('right_eye'), 
             self.keypoints.index('nose')],
            [self.keypoints.index('right_eye'), 
             self.keypoints.index('right_ear')],
            [self.keypoints.index('left_eye'), 
             self.keypoints.index('nose')],
            [self.keypoints.index('left_eye'), 
             self.keypoints.index('left_ear')],
            [self.keypoints.index('right_shoulder'), 
             self.keypoints.index('right_elbow')],
            [self.keypoints.index('right_elbow'), 
             self.keypoints.index('right_wrist')],
            [self.keypoints.index('left_shoulder'), 
             self.keypoints.index('left_elbow')],
            [self.keypoints.index('left_elbow'), 
             self.keypoints.index('left_wrist')],
            [self.keypoints.index('right_hip'), 
             self.keypoints.index('right_knee')],
            [self.keypoints.index('right_knee'), 
             self.keypoints.index('right_ankle')],
            [self.keypoints.index('left_hip'), 
             self.keypoints.index('left_knee')],
            [self.keypoints.index('left_knee'), 
             self.keypoints.index('left_ankle')],
            [self.keypoints.index('right_shoulder'), 
             self.keypoints.index('left_shoulder')],
            [self.keypoints.index('right_hip'), 
             self.keypoints.index('left_hip')],
            [self.keypoints.index('right_shoulder'), 
             self.keypoints.index('right_hip')],
            [self.keypoints.index('left_shoulder'), 
             self.keypoints.index('left_hip')],
        ]
        
    def inference(self, img_name):
        """KeypointRCNN_ResNet50_FPN inference

        Args:
            img_name (str or Path): image name

        Returns:
            numpy.ndarrays: boxes, labels, scores, keypoints, keypoints_scores
            as for PyTorch keypointrcnn_resnet50_fpn model
        """
        # Check if file exists 
        if Path(img_name).is_file():
            image = cv2.imread(str(img_name))
            self.image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), Path(img_name))
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = (keypointrcnn_resnet50_fpn(weights='DEFAULT')
                 .eval().to(device))
        transforms_ = transforms.ToTensor()
        img_tensor = transforms_(image).to(device)
        with torch.no_grad():
            predictions = model([img_tensor])
        
        # Model predictions to numpy arrays
        (self.boxes, self.labels, self.confs, 
         self.all_keypoints, self.all_scores) = (
            [x.cpu().numpy() for x in predictions[0].values()])
        
        # GPU memory cleansing
        del model, img_tensor, predictions
        torch.cuda.empty_cache()
                
        return (self.boxes, self.labels, self.confs, 
                self.all_keypoints, self.all_scores)
    
    def draw(self, keypoint_threshold=2, conf_threshold=0.9):
        """Draw a skeleton

        Args:
            keypoint_threshold (int, optional): keypoint detection threshold. 
                                                Defaults to 2.
            conf_threshold (float, optional): instance detection threshold. 
                                                Defaults to 0.9.
        """
        # image to draw skeleton
        self.skeleton_img = self.image.copy()    
    
        # color sequence 
        cmap = plt.get_cmap('jet')
        color_ids = (np.linspace(0, 255, len(self.confs) + 2)
                 .astype(int).tolist()[1:-1])
        
        # for each instance
        for person_id, conf in enumerate(self.confs): 
            if conf > conf_threshold:
                keypoints = self.all_keypoints[person_id, ...]
                scores = self.all_scores[person_id, ...]
                color = tuple(
                    np.asarray(cmap(color_ids[person_id])[:-1]) * 255)
                
                # for each keepoint
                for kp_id, score in enumerate(scores):
                    if score > keypoint_threshold:
                        keypoint = tuple(map(int, keypoints[kp_id, :2]))
                        cv2.circle(self.skeleton_img, keypoint, 5, color, -1)
                          
                # for each joint
                for joint in self.joints:
                    kp1_id,  kp2_id = joint[0], joint[1]
                    if (scores[kp1_id] > keypoint_threshold 
                        and  scores[kp2_id] > keypoint_threshold
                    ):
                        keypoint1 = tuple(map(int, keypoints[kp1_id, :2]))
                        keypoint2 = tuple(map(int, keypoints[kp2_id, :2]))
                        cv2.line(self.skeleton_img, keypoint1, keypoint2,
                                 color, thickness=2)
        
        fig = plt.figure(figsize=(16, 10))
        plt.imshow(self.skeleton_img);
        
        return
