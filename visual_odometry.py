"""
* This file is part of PYSLAM 
*
* Copyright (C) 2016-present Luigi Freda <luigi dot freda at gmail dot com> 
*
* PYSLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* PYSLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with PYSLAM. If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np 
import cv2
import platform 
from enum import Enum

from feature_tracker import FeatureTrackerTypes, FeatureTrackingResult, FeatureTracker
from utils_geom import poseRt, is_rotation_matrix, closest_rotation_matrix
from timer import TimerFps

class VoStage(Enum):
    NO_IMAGES_YET   = 0     # no image received 
    GOT_FIRST_IMAGE = 1     # got first image, we can proceed in a normal way (match current image with previous image)
    
kVerbose=True     
kMinNumFeature = 2000
kRansacThresholdNormalized = 0.0004  # metric threshold used for normalized image coordinates (originally 0.0003)
kRansacThresholdPixels = 0.1         # pixel threshold used for image coordinates 
kAbsoluteScaleThreshold = 0.1        # absolute translation scale; it is also the minimum translation norm for an accepted motion 
kUseEssentialMatrixEstimation = True # using the essential matrix fitting algorithm is more robust RANSAC given five-point algorithm solver 
kRansacProb = 0.999                  # (originally 0.999)
kUseGroundTruthScale = True 
 
''' 
// q: Explain the logic of the VisualOdometry class to a non-expert.
// a: The VisualOdometry class is a class that estimates the camera pose in a sequence of images. 
    It uses a feature tracker to track features between frames and estimate the camera pose. 
    The class keeps track of the current and previous images, the keypoints and descriptors of the features in the images, 
    and the camera rotation and translation. It also uses ground truth data to estimate the scale of the translation. The class has methods
    to process the first frame, track features in subsequent frames, estimate the camera pose, and update the history of poses and translations.
    The class also has methods to draw the feature tracks on the images and manage the stage of processing the images.
    The VisualOdometry class is a key component in the visual odometry pipeline, which is used in SLAM systems to estimate the camera trajectory
    and map the environment.

    // q: Explain about the scale in case where there is no ground truth available. Is it addressed in the code?
    // a: In the code, the scale is estimated using the average pixel shift between the matched keypoints in consecutive frames.
    If the average pixel shift is greater than a threshold and the absolute scale is greater than a threshold, the estimated translation is updated
    using the ground truth scale. This approach helps to maintain a coherent trajectory by adjusting the translation based on the estimated scale.
    However, without ground truth data, the scale estimation may not be accurate, and the trajectory may drift over time.
    This issue is not fully addressed in the code, as it relies on ground truth data for scale estimation.

// q: List of methods in the VisualOdometry class and their purpose.
// a: The VisualOdometry class has the following methods:
    - __init__: Initializes the VisualOdometry object with the camera model, ground truth data, and feature tracker.    
    - getAbsoluteScale: Computes the absolute scale of the translation using ground truth data.
    - computeFundamentalMatrix: Computes the fundamental matrix between two sets of keypoints.
    - removeOutliersByMask: Removes outliers from the matched keypoints based on a mask.
    - estimatePose: Estimates the camera pose between two frames using the essential matrix.
    - processFirstFrame: Processes the first frame by detecting and computing features.
    - processFrame: Processes a frame by tracking features, estimating the camera pose, and updating the keypoints history.
    - track: Tracks features in a frame and updates the camera pose estimation.
    - drawFeatureTracks: Draws the feature tracks on the image.
    - updateHistory: Updates the history of poses and translations based on the estimated and ground truth data.

// q: Explain the logic of the fundamental matrix computation and the essential matrix estimation - math behind it.
// a: The fundamental matrix is a 3x3 matrix that relates corresponding points in two images taken by a pinhole camera.
    It represents the epipolar geometry between the two images and is used to find the epipolar lines for feature matching.
    The fundamental matrix is computed using the eight-point algorithm or the normalized eight-point algorithm.
    The essential matrix is a 3x3 matrix that relates the camera poses between two images. It is derived from the fundamental matrix
    and the camera intrinsic parameters. The essential matrix is used to estimate the relative camera pose between two frames.  
    The essential matrix is computed using the fundamental matrix and the camera intrinsic matrix as E = K.T * F * K, where K is the camera intrinsic matrix.
    The essential matrix is decomposed into the rotation and translation components using the singular value decomposition (SVD) method.
    The rotation matrix R and translation vector t are estimated up to scale, and the scale is extracted from the ground truth data to recover the correct translation.

// q: Explain the logic of the camera pose estimation and the use of the essential matrix in the VisualOdometry class.
// a: The camera pose estimation in the VisualOdometry class is based on the essential matrix fitting algorithm, which uses the five-point algorithm solver by D. Nister.

// q: Explaim SVD in the context of camera pose estimation.
// a: Singular Value Decomposition (SVD) is a mathematical method used to decompose a matrix into three matrices: U, S, and V.
    In the context of camera pose estimation, SVD is used to decompose the essential matrix into the rotation and translation components.
    The essential matrix E is decomposed as E = U * S * V.T, where U and V are orthogonal matrices and S is a diagonal matrix with singular values.
    The rotation matrix R and translation vector t are extracted from the U and V matrices to estimate the camera pose between two frames.

// q: Give R and t in terms of the essential matrix E and its decomposition.
// a: The essential matrix E is decomposed as E = U * S * V.T, where U and V are orthogonal matrices and S is a diagonal matrix with singular values.
    A singular value is a scalar value that represents the scaling factor of the corresponding singular vector.
    The rotation matrix R and translation vector t are extracted from the U and V matrices as follows:
    - R = U * W * V.T, where W = [[0, -1, 0], [1, 0, 0], [0, 0, 1]] or [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]
    - t = u3, where u3 is the third column of the U matrix.

'''

# This class is a first start to understand the basics of inter frame feature tracking and camera pose estimation.
# It combines the simplest VO ingredients without performing any image point triangulation or 
# windowed bundle adjustment. At each step $k$, it estimates the current camera pose $C_k$ with respect to the previous one $C_{k-1}$. 
# The inter frame pose estimation returns $[R_{k-1,k},t_{k-1,k}]$ with $||t_{k-1,k}||=1$. 
# With this very basic approach, you need to use a ground truth in order to recover a correct inter-frame scale $s$ and estimate a 
# valid trajectory by composing $C_k = C_{k-1} * [R_{k-1,k}, s t_{k-1,k}]$. 
class VisualOdometry(object):
    def __init__(self, cam, groundtruth, feature_tracker : FeatureTracker):
        self.stage = VoStage.NO_IMAGES_YET
        self.cam = cam
        self.cur_image = None   # current image
        self.prev_image = None  # previous/reference image

        self.kps_ref = None  # reference keypoints 
        self.des_ref = None # refeference descriptors 
        self.kps_cur = None  # current keypoints 
        self.des_cur = None # current descriptors 

        self.cur_R = np.eye(3,3) # current rotation 
        self.cur_t = np.zeros((3,1)) # current translation 

        self.trueX, self.trueY, self.trueZ = None, None, None
        self.groundtruth = groundtruth
        
        self.feature_tracker = feature_tracker
        self.track_result = None 

        self.mask_match = None # mask of matched keypoints used for drawing 
        self.draw_img = None 

        self.init_history = True 
        self.poses = []              # history of poses
        self.t0_est = None           # history of estimated translations      
        self.t0_gt = None            # history of ground truth translations (if available)
        self.traj3d_est = []         # history of estimated translations centered w.r.t. first one
        self.traj3d_gt = []          # history of estimated ground truth translations centered w.r.t. first one     

        self.num_matched_kps = None    # current number of matched keypoints  
        self.num_inliers = None        # current number of inliers 

        self.timer_verbose = False # set this to True if you want to print timings 
        self.timer_main = TimerFps('VO', is_verbose = self.timer_verbose)
        self.timer_pose_est = TimerFps('PoseEst', is_verbose = self.timer_verbose)
        self.timer_feat = TimerFps('Feature', is_verbose = self.timer_verbose)

    # get current translation scale from ground-truth if groundtruth is not None 
    def getAbsoluteScale(self, frame_id):  
        if self.groundtruth is not None and kUseGroundTruthScale:
            self.trueX, self.trueY, self.trueZ, scale = self.groundtruth.getPoseAndAbsoluteScale(frame_id)
            return scale
        else:
            self.trueX = 0 
            self.trueY = 0 
            self.trueZ = 0
            return 1

    def computeFundamentalMatrix(self, kps_ref, kps_cur):
            F, mask = cv2.findFundamentalMat(kps_ref, kps_cur, cv2.FM_RANSAC, kRansacThresholdPixels, kRansacProb)
            if F is None or F.shape == (1, 1):
                # no fundamental matrix found
                raise Exception('No fundamental matrix found')
            elif F.shape[0] > 3:
                # more than one matrix found, just pick the first
                F = F[0:3, 0:3]
            return np.matrix(F), mask 	

    def removeOutliersByMask(self, mask): 
        if mask is not None:    
            n = self.kpn_cur.shape[0]     
            mask_index = [ i for i,v in enumerate(mask) if v > 0]    
            self.kpn_cur = self.kpn_cur[mask_index]           
            self.kpn_ref = self.kpn_ref[mask_index]           
            if self.des_cur is not None: 
                self.des_cur = self.des_cur[mask_index]        
            if self.des_ref is not None: 
                self.des_ref = self.des_ref[mask_index]  
            if kVerbose:
                print('removed ', n-self.kpn_cur.shape[0],' outliers')                

    # fit essential matrix E with RANSAC such that:  p2.T * E * p1 = 0  where  E = [t21]x * R21
    # out: [Rrc, trc]   (with respect to 'ref' frame) 
    # NB means Nota Bene (Note Well) - historically used to draw attention to something important
    # N.B.1: trc is estimated up to scale (i.e. the algorithm always returns ||trc||=1, we need a scale in order to recover a translation which is coherent with previous estimated poses)
    # N.B.2: this function has problems in the following cases: [see Hartley/Zisserman Book]
    # - 'geometrical degenerate correspondences', e.g. all the observed features lie on a plane (the correct model for the correspondences is an homography) or lie on a ruled quadric 
    # - degenerate motions such a pure rotation (a sufficient parallax is required) or an infinitesimal viewpoint change (where the translation is almost zero)
    # N.B.3: the five-point algorithm (used for estimating the Essential Matrix) seems to work well in the degenerate planar cases [Five-Point Motion Estimation Made Easy, Hartley]
    # N.B.4: as reported above, in case of pure rotation, this algorithm will compute a useless fundamental matrix which cannot be decomposed to return the rotation 
    def estimatePose(self, kps_ref, kps_cur):	
        kp_ref_u = self.cam.undistort_points(kps_ref)	
        kp_cur_u = self.cam.undistort_points(kps_cur)	        
        self.kpn_ref = self.cam.unproject_points(kp_ref_u)
        self.kpn_cur = self.cam.unproject_points(kp_cur_u)
        if kUseEssentialMatrixEstimation:
            ransac_method = None 
            try: 
                ransac_method = cv2.USAC_MSAC 
            except: 
                ransac_method = cv2.RANSAC
            # the essential matrix algorithm is more robust since it uses the five-point algorithm solver by D. Nister (see the notes and paper above )
            E, self.mask_match = cv2.findEssentialMat(self.kpn_cur, self.kpn_ref, focal=1, pp=(0., 0.), method=ransac_method, prob=kRansacProb, threshold=kRansacThresholdNormalized)
        else:
            # just for the hell of testing fundamental matrix fitting ;-) 
            F, self.mask_match = self.computeFundamentalMatrix(kp_cur_u, kp_ref_u)
            E = self.cam.K.T @ F @ self.cam.K    # E = K.T * F * K 
        #self.removeOutliersFromMask(self.mask)  # do not remove outliers, the last unmatched/outlier features can be matched and recognized as inliers in subsequent frames                          
        _, R, t, mask = cv2.recoverPose(E, self.kpn_cur, self.kpn_ref, focal=1, pp=(0., 0.))   
        return R,t  # Rrc, trc (with respect to 'ref' frame) 		

    def processFirstFrame(self):
        # only detect on the current image 
        # Here kp means keypoints and des means descriptors, ref means reference frame
        self.kps_ref, self.des_ref = self.feature_tracker.detectAndCompute(self.cur_image)
        # convert from list of keypoints to an array of points 
        self.kps_ref = np.array([x.pt for x in self.kps_ref], dtype=np.float32) if self.kps_ref is not None else None
        self.draw_img = self.drawFeatureTracks(self.cur_image)

    def processFrame(self, frame_id):
        # track features 
        self.timer_feat.start()
        self.track_result = self.feature_tracker.track(self.prev_image, self.cur_image, self.kps_ref, self.des_ref)
        self.timer_feat.refresh()
        # estimate pose 
        self.timer_pose_est.start()
        R, t = self.estimatePose(self.track_result.kps_ref_matched, self.track_result.kps_cur_matched)     
        self.timer_pose_est.refresh()
        # update keypoints history  
        self.kps_ref = self.track_result.kps_ref ##ere updating the reference keypoints with the current keypoints
        self.kps_cur = self.track_result.kps_cur
        self.des_cur = self.track_result.des_cur 
        self.num_matched_kps = self.kpn_ref.shape[0] 
        self.num_inliers =  np.sum(self.mask_match)
        #compute average delta pixel shift
        self.average_pixel_shift = np.mean(np.abs(self.track_result.kps_ref_matched - self.track_result.kps_cur_matched))
        print(f'average pixel shift: {self.average_pixel_shift}')
        if kVerbose:        
            print('# matched points: ', self.num_matched_kps, ', # inliers: ', self.num_inliers, ', matcher type: ', self.feature_tracker.matcher.matcher_type.name if self.feature_tracker.matcher is not None else 'None', ', tracker type: ', self.feature_tracker.tracker_type.name)      
        # t is estimated up to scale (i.e. the algorithm always returns ||trc||=1, we need a scale in order to recover a translation which is coherent with the previous estimated ones)
        absolute_scale = self.getAbsoluteScale(frame_id)
        if(absolute_scale > kAbsoluteScaleThreshold and self.average_pixel_shift > 1):
            # compose absolute motion [Rwa,twa] with estimated relative motion [Rab,s*tab] (s is the scale extracted from the ground truth)
            # [Rwb,twb] = [Rwa,twa]*[Rab,tab] = [Rwa*Rab|twa + Rwa*tab]
            print('estimated t with norm |t|: ', np.linalg.norm(t), ' (just for sake of clarity)')
            self.cur_t = self.cur_t + absolute_scale*self.cur_R.dot(t) 
            self.cur_R = self.cur_R.dot(R)
            if not is_rotation_matrix(self.cur_R):
                print(f'Correcting rotation matrix: {self.cur_R}')
                self.cur_R = closest_rotation_matrix(self.cur_R)
        # draw image         
        self.draw_img = self.drawFeatureTracks(self.cur_image) 
        # check if we have enough features to track otherwise detect new ones and start tracking from them (used for LK tracker) 
        if (self.feature_tracker.tracker_type == FeatureTrackerTypes.LK) and (self.kps_ref.shape[0] < self.feature_tracker.num_features): 
            self.kps_cur, self.des_cur = self.feature_tracker.detectAndCompute(self.cur_image)           
            self.kps_cur = np.array([x.pt for x in self.kps_cur], dtype=np.float32) # convert from list of keypoints to an array of points   
            if kVerbose:     
                print('# new detected points: ', self.kps_cur.shape[0])                  
        self.kps_ref = self.kps_cur
        self.des_ref = self.des_cur
        self.updateHistory()           
        

    def track(self, img, frame_id):
        if kVerbose:
            print('..................................')
            print('frame: ', frame_id) 
        # convert image to gray if needed    
        if img.ndim>2:
            img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)             
        # check coherence of image size with camera settings 
        assert(img.ndim==2 and img.shape[0]==self.cam.height and img.shape[1]==self.cam.width), "Frame: provided image has not the same size as the camera model or image is not grayscale"
        self.cur_image = img
        # manage and check stage 
        if(self.stage == VoStage.GOT_FIRST_IMAGE):
            self.processFrame(frame_id)
        elif(self.stage == VoStage.NO_IMAGES_YET):
            self.processFirstFrame()
            self.stage = VoStage.GOT_FIRST_IMAGE            
        self.prev_image = self.cur_image    
        # update main timer (for profiling)
        self.timer_main.refresh()  
  

    def drawFeatureTracks(self, img, reinit = False):
        draw_img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        num_outliers = 0        
        if(self.stage == VoStage.GOT_FIRST_IMAGE):            
            if reinit:
                for p1 in self.kps_cur:
                    a,b = p1.ravel()
                    cv2.circle(draw_img,(a,b),1, (0,255,0),-1)                    
            else:    
                for i,pts in enumerate(zip(self.track_result.kps_ref_matched, self.track_result.kps_cur_matched)):
                    drawAll = False # set this to true if you want to draw outliers 
                    if self.mask_match[i] or drawAll:
                        p1, p2 = pts 
                        a,b = p1.astype(int).ravel()
                        c,d = p2.astype(int).ravel()
                        cv2.line(draw_img, (a,b),(c,d), (0,255,0), 1)
                        cv2.circle(draw_img,(a,b),1, (0,0,255),-1)   
                    else:
                        num_outliers+=1
            if kVerbose:
                print('# outliers: ', num_outliers)     
        return draw_img            

    def updateHistory(self):
        if (self.init_history is True) and (self.trueX is not None):
            self.t0_est = np.array([self.cur_t[0], self.cur_t[1], self.cur_t[2]])  # starting translation 
            self.t0_gt  = np.array([self.trueX, self.trueY, self.trueZ])           # starting translation 
            self.init_history = False 
        if (self.t0_est is not None) and (self.t0_gt is not None):             
            p = [self.cur_t[0]-self.t0_est[0], self.cur_t[1]-self.t0_est[1], self.cur_t[2]-self.t0_est[2]]   # the estimated traj starts at 0
            self.traj3d_est.append(p)
            pg = [self.trueX-self.t0_gt[0], self.trueY-self.t0_gt[1], self.trueZ-self.t0_gt[2]]  # the groudtruth traj starts at 0  
            self.traj3d_gt.append(pg)     
            self.poses.append(poseRt(self.cur_R, p))  
            #self.poses.append(poseRt(self.cur_R, p[0])) 
