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

import sys 
import math 
import numpy as np
import cv2 

from frame import Frame, FrameShared, are_map_points_visible, are_map_points_visible_in_frame
from keyframe import KeyFrame
from map_point import MapPoint, predict_detection_levels

from utils_geom import skew, add_ones, normalize_vector, computeF12, check_dist_epipolar_line, Sim3Pose
from utils_draw import draw_lines, draw_points 
from utils_sys import Printer, getchar
from config_parameters import Parameters  
from timer import Timer
from rotation_histogram import RotationHistogram


kMinDistanceFromEpipole = Parameters.kMinDistanceFromEpipole
kMinDistanceFromEpipole2 = kMinDistanceFromEpipole*kMinDistanceFromEpipole
kCheckFeaturesOrientation = Parameters.kCheckFeaturesOrientation 


# propagate map point matches from f_ref to f_cur (access frames from tracking thread, no need to lock)
def propagate_map_point_matches(f_ref, f_cur, idxs_ref, idxs_cur,
                                max_descriptor_distance=None):
    if max_descriptor_distance is None:
        max_descriptor_distance = Parameters.kMaxDescriptorDistance
        
    idx_ref_out = []
    idx_cur_out = []
    
    rot_histo = RotationHistogram()
    check_orientation = kCheckFeaturesOrientation and FrameShared.oriented_features
        
    # populate f_cur with map points by propagating map point matches of f_ref; 
    # to this aim, we use map points observed in f_ref and keypoint matches between f_ref and f_cur  
    num_matched_map_pts = 0
    for i, idx in enumerate(idxs_ref): # iterate over keypoint matches 
        p_ref = f_ref.points[idx]
        if p_ref is None: # we don't have a map point P for i-th matched keypoint in f_ref
            continue 
        if f_ref.outliers[idx] or p_ref.is_bad: # do not consider pose optimization outliers or bad points 
            continue  
        idx_cur = idxs_cur[i]
        p_cur = f_cur.points[idx_cur]
        if p_cur is not None: # and p_cur.num_observations > 0: # if we already matched p_cur => no need to propagate anything  
            continue
        des_distance = p_ref.min_des_distance(f_cur.des[idx_cur])
        if des_distance > max_descriptor_distance: 
            continue 
        if p_ref.add_frame_view(f_cur, idx_cur): # => P is matched to the i-th matched keypoint in f_cur
            num_matched_map_pts += 1
            idx_ref_out.append(idx)
            idx_cur_out.append(idx_cur)
            
            if check_orientation:
                index_match = len(idx_cur_out)-1
                rot = f_ref.angles[idx]-f_cur.angles[idx_cur]
                rot_histo.push(rot, index_match)
            
    if check_orientation:            
        valid_match_idxs = rot_histo.get_valid_idxs()     
        print('checking orientation consistency - valid matches % :', len(valid_match_idxs)/max(1,len(idxs_cur))*100,'% of ', len(idxs_cur),'matches')
        #print('rotation histogram: ', rot_histo)
        idx_ref_out = np.array(idx_ref_out)[valid_match_idxs]
        idx_cur_out = np.array(idx_cur_out)[valid_match_idxs]
        num_matched_map_pts = len(valid_match_idxs)            
                            
    return num_matched_map_pts, idx_ref_out, idx_cur_out  

  
# search by projection matches between {map points of f_ref} and {keypoints of f_cur},  (access frames from tracking thread, no need to lock)
def search_frame_by_projection(f_ref: Frame, f_cur: Frame,
                               max_reproj_distance=Parameters.kMaxReprojectionDistanceFrame,
                               max_descriptor_distance=None,
                               ratio_test=Parameters.kMatchRatioTestMap,
                               is_monocular=True,
                               already_matched_ref_idxs=None):
    if max_descriptor_distance is None:
        max_descriptor_distance = Parameters.kMaxDescriptorDistance    

    found_pts_count = 0
    idxs_ref = []
    idxs_cur = [] 
    
    rot_histo = RotationHistogram()
    check_orientation = kCheckFeaturesOrientation and FrameShared.oriented_features    
    
    trc = None
    forward = False
    backward = False
    if not is_monocular:
        Tcw = f_cur.pose
        Rcw = Tcw[:3,:3]
        tcw = Tcw[:3,3]
        twc = -Rcw.T.dot(tcw)
        
        Trw = f_ref.pose
        Rrw = Trw[:3,:3]
        trw = Trw[:3,3]
        trc = Rrw.T.dot(twc)+trw
        forward = trc[2] > f_cur.camera.b
        backward = trc[2] < -f_cur.camera.b
          
    # get all matched points of f_ref which are non-outlier 
    matched_ref_idxs = np.flatnonzero( (f_ref.points!=None) & (f_ref.outliers==False)) 
    
    # if we have some already matched points in reference frame, remove them from the list
    if already_matched_ref_idxs is not None:
        matched_ref_idxs = np.setdiff1d(matched_ref_idxs, already_matched_ref_idxs)
    
    matched_ref_points = f_ref.points[matched_ref_idxs]

    # project f_ref points on frame f_cur
    projs, depths = f_cur.project_map_points(matched_ref_points, f_cur.camera.is_stereo())
    # check if points lie on the image frame 
    is_visible = f_cur.are_in_image(projs, depths)

    # # check if points are visible 
    # is_visible, projs, depths, dists = f_cur.are_visible(matched_ref_points)
        
    kp_ref_octaves = f_ref.octaves[matched_ref_idxs]       
    kp_ref_scale_factors = FrameShared.feature_manager.scale_factors[kp_ref_octaves]              
    radiuses = max_reproj_distance * kp_ref_scale_factors     
    kd_cur_idxs = f_cur.kd.query_ball_point(projs[:,:2], radiuses)   
    
    do_check_stereo_reproj_err = f_cur.kps_ur is not None 
                     
    for ref_idx,p,j in zip(matched_ref_idxs, matched_ref_points, range(len(matched_ref_points))):
    
        if not is_visible[j]:
            continue 
        
        kp_ref_octave = f_ref.octaves[ref_idx]  
        
        best_dist = math.inf 
        #best_dist2 = math.inf
        best_level = -1 
        #best_level2 = -1                   
        best_k_idx = -1   
        best_ref_idx = -1          
                  
        kd_cur_idxs_j = kd_cur_idxs[j]                
        if do_check_stereo_reproj_err:
            check_stereo = f_cur.kps_ur[kd_cur_idxs_j]>0 
            kp_cur_octaves = f_cur.octaves[kd_cur_idxs_j]       
            kp_cur_scale_factors = FrameShared.feature_manager.scale_factors[kp_cur_octaves]  
            errs_ur = np.fabs(projs[j,2] - f_cur.kps_ur[kd_cur_idxs_j]) 
            ok_errs_ur = np.where(check_stereo, errs_ur < max_reproj_distance * kp_cur_scale_factors,True)  
                           
        for h, kd_idx in enumerate(kd_cur_idxs[j]):                   
            
            p_f_cur = f_cur.points[kd_idx]
            if  p_f_cur is not None:
                if p_f_cur.num_observations > 0: # we already matched p_f_cur => discard it 
                    continue          
    
            p_f_cur_octave = f_cur.octaves[kd_idx]
            if is_monocular:
                if p_f_cur_octave < (kp_ref_octave-1) or p_f_cur_octave > (kp_ref_octave+1):
                    continue
            else:
                if backward and p_f_cur_octave > kp_ref_octave:
                    continue
                elif forward and p_f_cur_octave < kp_ref_octave:
                    continue
                elif p_f_cur_octave < (kp_ref_octave-1) or p_f_cur_octave > (kp_ref_octave+1):
                    continue
                                       
            if do_check_stereo_reproj_err:
                if not ok_errs_ur[h]:
                    continue
                    
            descriptor_dist = p.min_des_distance(f_cur.des[kd_idx])
            if descriptor_dist < best_dist:                
                best_dist = descriptor_dist
                best_k_idx = kd_idx
                best_ref_idx = ref_idx     
                
            # if descriptor_dist < best_dist:                                      
            #     best_dist2 = best_dist
            #     best_level2 = best_level
            #     best_dist = descriptor_dist
            #     best_level = f_cur.octaves[kd_idx]
            #     best_k_idx = kd_idx  
            #     best_ref_idx = i                      
            # else: 
            #     if descriptor_dist < best_dist2:  
            #         best_dist2 = descriptor_dist
            #         best_level2 = f_cur.octaves[kd_idx]                       
                                
        #if best_k_idx > -1 and best_dist < max_descriptor_distance:
        if best_dist < max_descriptor_distance: 
            # apply match distance ratio test only if the best and second are in the same scale level 
            #if (best_level2 == best_level) and (best_dist > best_dist2 * ratio_test): 
            #    continue                        
            #print('b_dist : ', best_dist)            
            if p.add_frame_view(f_cur, best_k_idx):
                found_pts_count += 1
                idxs_ref.append(best_ref_idx)
                idxs_cur.append(best_k_idx)   
                
                if check_orientation:                  
                    index_match = len(idxs_cur)-1
                    rot = f_ref.angles[best_ref_idx]-f_cur.angles[best_k_idx]
                    rot_histo.push(rot,index_match)
                
            #print('best des distance: ', best_dist, ", max dist: ", max_descriptor_distance)                                        
            #des_dists.append(best_dist)
            
    if check_orientation:            
        valid_match_idxs = rot_histo.get_valid_idxs()     
        print('checking orientation consistency - valid matches % :', len(valid_match_idxs)/max(1,len(idxs_cur))*100,'% of ', len(idxs_cur),'matches')
        #print('rotation histogram: ', rot_histo)
        idxs_ref = np.array(idxs_ref)[valid_match_idxs]
        idxs_cur = np.array(idxs_cur)[valid_match_idxs]
        found_pts_count = len(valid_match_idxs)
    
    return np.array(idxs_ref), np.array(idxs_cur), found_pts_count
    #return idxs_ref, idxs_cur, found_pts_count 


# search by projection matches between {input map points} and {unmatched keypoints of frame f_cur}, (access frame from tracking thread, no need to lock)
def search_map_by_projection(points, f_cur: Frame, 
                             max_reproj_distance=Parameters.kMaxReprojectionDistanceMap, 
                             max_descriptor_distance=None,
                             ratio_test=Parameters.kMatchRatioTestMap):
    if max_descriptor_distance is None:
        max_descriptor_distance = Parameters.kMaxDescriptorDistance  
           
    found_pts_count = 0
    found_pts_fidxs = []   # idx of matched points in current frame 
    
    #reproj_dists = []
    
    if len(points) == 0:
        return 0 
            
    # check if points are visible 
    visible_pts, projs, depths, dists = f_cur.are_visible(points)
    
    predicted_levels = predict_detection_levels(points, dists) 
    kp_scale_factors = FrameShared.feature_manager.scale_factors[predicted_levels]              
    radiuses = max_reproj_distance * kp_scale_factors     
    kd_cur_idxs = f_cur.kd.query_ball_point(projs, radiuses)
                           
    for i, p in enumerate(points):
        if not visible_pts[i] or p.is_bad:     # point not visible in frame or is bad 
            continue        
        if p.last_frame_id_seen == f_cur.id:   # we already matched this map point to current frame or it was outlier 
            continue
        
        p.increase_visible()
          
        predicted_level = predicted_levels[i]         
                       
        best_dist = math.inf 
        best_dist2 = math.inf
        best_level = -1 
        best_level2 = -1               
        best_k_idx = -1  

        # find closest keypoints of f_cur         
        for kd_idx in kd_cur_idxs[i]:
     
            p_f = f_cur.points[kd_idx]
            # check there is not already a match               
            if  p_f is not None:
                if p_f.num_observations > 0:
                    continue 
                
            # check detection level     
            kp_level = f_cur.octaves[kd_idx]    
            if (kp_level<predicted_level-1) or (kp_level>predicted_level):
                continue                
                
            descriptor_dist = p.min_des_distance(f_cur.des[kd_idx])
  
            if descriptor_dist < best_dist:                                      
                best_dist2 = best_dist
                best_level2 = best_level
                best_dist = descriptor_dist
                best_level = kp_level
                best_k_idx = kd_idx    
            else: 
                if descriptor_dist < best_dist2:  
                    best_dist2 = descriptor_dist
                    best_level2 = kp_level                                        
                                                       
        #if best_k_idx > -1 and best_dist < max_descriptor_distance:
        if best_dist < max_descriptor_distance:            
            # apply match distance ratio test only if the best and second are in the same scale level 
            if (best_level2 == best_level) and (best_dist > best_dist2 * ratio_test): 
                continue 
            #print('best des distance: ', best_dist, ", max dist: ", Parameters.kMaxDescriptorDistance)                    
            if p.add_frame_view(f_cur, best_k_idx):
                found_pts_count += 1  
                found_pts_fidxs.append(best_k_idx)  
            
            #reproj_dists.append(np.linalg.norm(projs[i] - f_cur.kpsu[best_k_idx]))   
            
    # if len(reproj_dists) > 1:        
    #     reproj_dist_sigma = 1.4826 * np.median(reproj_dists)    
    # else:
    reproj_dist_sigma = max_descriptor_distance
                       
    return found_pts_count, reproj_dist_sigma, found_pts_fidxs   


# search by projection matches between {map points of last frames} and {unmatched keypoints of f_cur}, (access frame from tracking thread, no need to lock)
def search_local_frames_by_projection(map, f_cur, local_window=Parameters.kLocalBAWindow, max_descriptor_distance=None):
    if max_descriptor_distance is None:
        max_descriptor_distance = Parameters.kMaxDescriptorDistance
        
    # take the points in the last N frame 
    points = []
    frames = map.keyframes[-local_window:]
    f_points = set([p for f in frames for p in f.get_points() if (p is not None)])
    print('searching %d map points' % len(points))
    return search_map_by_projection(points, f_cur, max_descriptor_distance=max_descriptor_distance)  


# search by projection matches between {all map points} and {unmatched keypoints of f_cur}
def search_all_map_by_projection(map, f_cur, max_descriptor_distance=None):
    if max_descriptor_distance is None:
        max_descriptor_distance = Parameters.kMaxDescriptorDistance  
           
    return search_map_by_projection(map.get_points(), f_cur, max_descriptor_distance=max_descriptor_distance)      



# search by projection more matches between {input map points} and {unmatched keypoints of frame f_cur}
# in: 
#   points: input map points
#   f_cur: current frame
#   f_cur_matched_points: matched points in current frame  (f_cur_matched_points[i] is the i-th map point matched on f_cur or None)   
#   Scw: suggested se3 or sim3 transformation
# The suggested transformation Scw (in se3 or sim3) is used in the search (instead of using the current frame pose)
def search_more_map_points_by_projection(points: set, 
                                    f_cur: Frame,
                                    f_cur_matched_points: list,  # f_cur_matched_points[i] is the i-th map point matched in f_cur or None
                                    Scw,
                                    max_reproj_distance=Parameters.kMaxReprojectionDistanceMap, 
                                    max_descriptor_distance=None,
                                    print_fun=None):
    if max_descriptor_distance is None:
        max_descriptor_distance = 0.5*Parameters.kMaxDescriptorDistance # more conservative check
        
    found_pts_count = 0
    if len(points) == 0:
        return found_pts_count, f_cur_matched_points
            
    assert(len(f_cur.points) == len(f_cur_matched_points))
    
    # extract from sim3 Scw=[s*Rcw, tcw; 0, 1] the corresponding se3 transformation Tcw=[Rcw, tcw/s]
    if isinstance(Scw, np.ndarray):
        sRcw = Scw[:3,:3]
        scw = math.sqrt(np.dot(sRcw[0,:3], sRcw[0,:3]))
        Rcw = sRcw / scw
        tcw = Scw[:3,3]/scw
    elif isinstance(Scw, Sim3Pose):
        scw = Scw.s
        Rcw = Scw.R
        tcw = Scw.t/ scw
    else: 
        raise TypeError("Unsupported type '{}' for Scw".format(type(Scw)))
    
    if not isinstance(points, set):
        points = set(points)
    target_points = points.difference([p for p in f_cur_matched_points if p is not None])    
    
    if len(target_points) == 0:
        if print_fun is not None:
            print_fun('search_more_map_points_by_projection: no target points available after difference')
        return found_pts_count, f_cur_matched_points
    
    # check if points are visible     
    visible_pts, projs, depths, dists = are_map_points_visible_in_frame(target_points, f_cur, Rcw, tcw)
    
    if print_fun is not None:
        print_fun(f'search_more_map_points_by_projection: #visible points: {len(visible_pts)}')
    
    predicted_levels = predict_detection_levels(target_points, dists) 
    kp_scale_factors = FrameShared.feature_manager.scale_factors[predicted_levels]              
    radiuses = max_reproj_distance * kp_scale_factors     
    kd_cur_idxs = f_cur.kd.query_ball_point(projs, radiuses)
    
    # num_failures_vis_or_bad = 0
    # num_failures_pf_is_not_none = 0
    # num_failures_kp_level = 0
    # num_failures_max_des_distance = 0
                               
    for i, p in enumerate(target_points):
        if not visible_pts[i] or p.is_bad:     # point not visible in frame or is bad 
            # num_failures_vis_or_bad +=1
            continue        
  
        predicted_level = predicted_levels[i]         
                       
        best_dist = math.inf 
        best_k_idx = -1  
 
        # find closest keypoints of f_cur        
        for kd_idx in kd_cur_idxs[i]:
     
            p_f = f_cur_matched_points[kd_idx]
            # check there is not already a match in f_cur_matched_points              
            if  p_f is not None:
                # num_failures_pf_is_not_none +=1
                continue 
                
            # check detection level     
            kp_level = f_cur.octaves[kd_idx]    
            if (kp_level<predicted_level-1) or (kp_level>predicted_level):
                # if print_fun is not None:                
                #     print_fun(f'search_more_map_points_by_projection: bad kp level: {kp_level},  predicted_level: {predicted_level}')
                # num_failures_kp_level += 1
                continue                
                
            descriptor_dist = p.min_des_distance(f_cur.des[kd_idx])
  
            if descriptor_dist < best_dist:                                      
                best_dist = descriptor_dist
                best_k_idx = kd_idx                               
                                                       
        if best_dist < max_descriptor_distance:
            f_cur_matched_points[best_k_idx] = p
            found_pts_count += 1
        # else:
        #     if print_fun is not None:
        #         print_fun(f'search_more_map_points_by_projection: bad best_des_distance: {best_dist}, max_descriptor_distance: {max_descriptor_distance}')
        #     num_failures_max_des_distance += 1
            
    # if print_fun is not None:
    #     print_fun(f'search_more_map_points_by_projection: num_failures_vis_or_bad: {num_failures_vis_or_bad}')
    #     print_fun(f'search_more_map_points_by_projection: num_failures_pf_is_not_none: {num_failures_pf_is_not_none}')
    #     print_fun(f'search_more_map_points_by_projection: num_failures_kp_level: {num_failures_kp_level}')
    #     print_fun(f'search_more_map_points_by_projection: num_failures_max_des_distance: {num_failures_max_des_distance}')
    
    return found_pts_count, f_cur_matched_points
            

# search keypoint matches (for triangulations) between f1 and f2
# search for matches between unmatched keypoints (without a corresponding map point)
# in input we have already some pose estimates for f1 and f2
def search_frame_for_triangulation(kf1, kf2, idxs1=None, idxs2=None, 
                                   max_descriptor_distance=None,
                                   is_monocular=True):
    if max_descriptor_distance is None:
        max_descriptor_distance = 0.5*Parameters.kMaxDescriptorDistance # more conservative check
           
    idxs2_out = []
    idxs1_out = []
    num_found_matches = 0
    img2_epi = None     

    if __debug__:
        timer = Timer()
        timer.start()

    O1w = kf1.Ow
    O2w = kf2.Ow
    # compute epipoles
    e1,_ = kf1.project_point(O2w)  # in first frame 
    e2,_ = kf2.project_point(O1w)  # in second frame  
    #print('e1: ', e1)
    #print('e2: ', e2)    
    
    baseline = np.linalg.norm(O1w-O2w) 

    # if the translation is too small we cannot triangulate 
    if not is_monocular:  # we assume the Inializer has been used for building the first map 
        baseline < kf2.camera.b
        return idxs1_out, idxs2_out, num_found_matches, img2_epi # EXIT
    else:    
        medianDepth = kf2.compute_points_median_depth()
        if medianDepth == -1:
            Printer.orange("search for triangulation: f2 with no points")        
            medianDepth = kf1.compute_points_median_depth()        
        ratioBaselineDepth = baseline/medianDepth
        if ratioBaselineDepth < Parameters.kMinRatioBaselineDepth:  
            Printer.orange("search for triangulation: impossible with too low ratioBaselineDepth!")
            return idxs1_out, idxs2_out, num_found_matches, img2_epi # EXIT        

    # compute the fundamental matrix between the two frames by using their estimated poses 
    F12, H21 = computeF12(kf1, kf2)

    if idxs1 is None or idxs2 is None:
        timerMatch = Timer()
        timerMatch.start()
        matching_result = FrameShared.feature_matcher.match(kf1.img, kf2.img, kf1.des, kf2.des)  
        idxs1, idxs2 = matching_result.idxs1, matching_result.idxs2      
        if __debug__:        
            print('search_frame_for_triangulation - matching - timer: ', timerMatch.elapsed())        
    
    rot_histo = RotationHistogram()
    check_orientation = kCheckFeaturesOrientation and FrameShared.oriented_features     
        
        
    level_sigmas2 = FrameShared.feature_manager.level_sigmas2
    scale_factors = FrameShared.feature_manager.scale_factors
    
    # check epipolar constraints 
    for i1,i2 in zip(idxs1,idxs2):
        if kf1.get_point_match(i1) is not None or kf2.get_point_match(i2) is not None: # we are searching for keypoint matches where both keypoints do not have a corresponding map point 
            #print('existing point on match')
            continue 
        
        descriptor_dist = FrameShared.descriptor_distance(kf1.des[i1], kf2.des[i2])
        if descriptor_dist > max_descriptor_distance:
            continue     
        
        kp1 = kf1.kpsu[i1]
        #kp1_scale_factor = scale_factors[kf1.octaves[i1]]
        #kp1_size = f1.sizes[i1]
        # discard points which are too close to the epipole            
        #if np.linalg.norm(kp1-e1) < Parameters.kMinDistanceFromEpipole * kp1_scale_factor:                 
        #if np.linalg.norm(kp1-e1) - kp1_size < Parameters.kMinDistanceFromEpipole:  # N.B.: this is too much conservative => it filters too much                                       
        #    continue   
        
        kp2 = kf2.kpsu[i2]
        kp2_scale_factor = scale_factors[kf2.octaves[i2]]        
        # kp2_size = f2.sizes[i2]        
        # discard points which are too close to the epipole            
        delta = kp2-e2        
        #if np.linalg.norm(delta) < Parameters.kMinDistanceFromEpipole * kp2_scale_factor:   
        if np.inner(delta,delta) < kMinDistanceFromEpipole2 * kp2_scale_factor:   # OR.            
        # #if np.linalg.norm(delta) - kp2_size < Parameters.kMinDistanceFromEpipole:  # N.B.: this is too much conservative => it filters too much                                                  
             continue           
        
        # check epipolar constraint         
        sigma2_kp2 = level_sigmas2[kf2.octaves[i2]]
        if check_dist_epipolar_line(kp1,kp2,F12,sigma2_kp2):
            idxs1_out.append(i1)
            idxs2_out.append(i2)
            
            if check_orientation:
                index_match = len(idxs1_out)-1
                rot = kf1.angles[i1]-kf2.angles[i2]
                rot_histo.push(rot,index_match)            
        #else:
        #    print('discarding point match non respecting epipolar constraint')
         
    if check_orientation:            
        valid_match_idxs = rot_histo.get_valid_idxs()     
        #print('checking orientation consistency - valid matches % :', len(valid_match_idxs)/max(1,len(idxs1_out))*100,'% of ', len(idxs1_out),'matches')
        #print('rotation histogram: ', rot_histo)
        idxs1_out = np.array(idxs1_out)[valid_match_idxs]
        idxs2_out = np.array(idxs2_out)[valid_match_idxs]
                 
    num_found_matches = len(idxs1_out)
             
    if __debug__:
        print('search_frame_for_triangulation - timer: ', timer.elapsed())

    return idxs1_out, idxs2_out, num_found_matches, img2_epi


# search by projection matches between {input map points} and {keyframe points} and fuse them if they are close enough
def search_and_fuse(points, keyframe: KeyFrame, 
                    max_reproj_distance=Parameters.kMaxReprojectionDistanceFuse,
                    max_descriptor_distance = None,
                    ratio_test=Parameters.kMatchRatioTestMap):    
    if max_descriptor_distance is None:
        max_descriptor_distance = 0.5*Parameters.kMaxDescriptorDistance # more conservative check
    
    fused_pts_count = 0
    if len(points) == 0:
        Printer.red('search_and_fuse - no points')        
        return fused_pts_count
        
    # get all matched points of keyframe 
    good_pts_idxs = np.flatnonzero(points!=None) 
    good_pts = points[good_pts_idxs] 
     
    if len(good_pts_idxs) == 0:
        Printer.red('search_and_fuse - no matched points')
        return fused_pts_count
    
    # check if points are visible 
    good_pts_visible, good_projs, good_depths, good_dists = keyframe.are_visible(good_pts, keyframe.camera.is_stereo())
    
    if np.sum(good_pts_visible) == 0:
        Printer.red('search_and_fuse - no visible points')
        return fused_pts_count   
    
    predicted_levels = predict_detection_levels(good_pts, good_dists) 
    kp_scale_factors = FrameShared.feature_manager.scale_factors[predicted_levels]              
    radiuses = max_reproj_distance * kp_scale_factors     
    
    kd_idxs = keyframe.kd.query_ball_point(good_projs[:,:2], radiuses)    
    
    do_check_stereo_reproj_err = keyframe.kps_ur is not None    

    #for i, p in enumerate(points):
    for i,p,j in zip(good_pts_idxs,good_pts,range(len(good_pts))):            
                
        if not good_pts_visible[j] or p.is_bad:     # point not visible in frame or point is bad 
            #print('p[%d] visible: %d, bad: %d' % (i, int(good_pts_visible[j]), int(p.is_bad))) 
            continue  
                  
        if p.is_in_keyframe(keyframe):    # we already matched this map point to this keyframe
            #print('p[%d] already in keyframe' % (i)) 
            continue
                   
        predicted_level = predicted_levels[j]     
            
        best_dist = math.inf 
        best_dist2 = math.inf
        best_level = -1 
        best_level2 = -1               
        best_kd_idx = -1        
            
        # find closest keypoints of frame        
        proj = good_projs[j]

        kd_idxs_j = kd_idxs[j]                
        if do_check_stereo_reproj_err:
            check_stereo = keyframe.kps_ur[kd_idxs_j]>0 
            errs_ur = proj[2] - keyframe.kps_ur[kd_idxs_j] # proj_ur - kp_ur
            errs_ur2 = errs_ur*errs_ur
            
        inv_level_sigmas2 = FrameShared.feature_manager.inv_level_sigmas2
                        
        for h, kd_idx in enumerate(kd_idxs_j):             
                
            # check detection level     
            kp_level = keyframe.octaves[kd_idx]    
            if (kp_level<predicted_level-1) or (kp_level>predicted_level):   
                #print('p[%d] wrong predicted level **********************************' % (i))                       
                continue
        
            # check the reprojection error     
            kp = keyframe.kpsu[kd_idx]
            invSigma2 = inv_level_sigmas2[kp_level]            
                                    
            err = proj[:2] - kp
            chi2 = np.dot(err,err)*invSigma2     
            if do_check_stereo_reproj_err and check_stereo[h]:
                chi2 += errs_ur2[h]*invSigma2 
                if chi2 > Parameters.kChi2Stereo: # chi-square 3 DOFs  (Hartley Zisserman pg 119)
                    #print('p[%d] big reproj err %f **********************************' % (i,chi2))
                    continue
            else:           
                if chi2 > Parameters.kChi2Mono: # chi-square 2 DOFs  (Hartley Zisserman pg 119)
                    #print('p[%d] big reproj err %f **********************************' % (i,chi2))
                    continue                  
                            
            descriptor_dist = p.min_des_distance(keyframe.des[kd_idx])
            #print('p[%d] descriptor_dist %f **********************************' % (i,descriptor_dist))            
            
            #if descriptor_dist < max_descriptor_distance and descriptor_dist < best_dist:     
            if descriptor_dist < best_dist:                                      
                best_dist2 = best_dist
                best_level2 = best_level
                best_dist = descriptor_dist
                best_level = kp_level
                best_kd_idx = kd_idx   
            elif descriptor_dist < best_dist2:  # N.O.
                best_dist2 = descriptor_dist       
                best_level2 = kp_level                                   
                                                            
        #if best_kd_idx > -1 and best_dist < max_descriptor_distance:
        if best_dist < max_descriptor_distance:         
            # apply match distance ratio test only if the best and second are in the same scale level 
            if (best_level2 == best_level) and (best_dist > best_dist2 * ratio_test):  # N.O.
                #print('p[%d] best_dist > best_dist2 * ratio_test **********************************' % (i))
                continue                
            p_keyframe = keyframe.get_point_match(best_kd_idx)
            # if there is already a map point replace it otherwise add a new point
            if p_keyframe is not None:
                # if not p_keyframe.is_bad:
                #     if p_keyframe.num_observations > p.num_observations:
                #         p.replace_with(p_keyframe)
                #     else:
                #         p_keyframe.replace_with(p)        
                p_keyframe_is_bad, p_keyframe_is_good_with_better_num_obs = p_keyframe.is_bad_and_is_good_with_min_obs(p.num_observations)
                if not p_keyframe_is_bad:
                    if p_keyframe_is_good_with_better_num_obs:
                        p.replace_with(p_keyframe)
                    else:
                        p_keyframe.replace_with(p)
            else:
                p.add_observation(keyframe, best_kd_idx) 
                #p.update_info()    # done outside!
            fused_pts_count += 1                  
    return fused_pts_count     



# search by projection matches between {input map points} and {keyframe points} and fuse them if they are close enough
# use suggested Scw to project
def search_and_fuse_for_loop_correction(keyframe: KeyFrame, 
                                        Scw, 
                                        points, 
                                        replace_points,
                                        max_reproj_distance=Parameters.kLoopClosingMaxReprojectionDistanceFuse,
                                        max_descriptor_distance = None):    
    if max_descriptor_distance is None:
        max_descriptor_distance = 0.5*Parameters.kMaxDescriptorDistance # more conservative check
    
    assert(len(points) == len(replace_points))
    
    fused_pts_count = 0
    if len(points) == 0:
        Printer.red('search_and_fuse - no points')        
        return replace_points
        
    # get all matched points of keyframe 
    good_pts_idxs = np.flatnonzero(points!=None) 
    good_pts = points[good_pts_idxs] 
     
    if len(good_pts_idxs) == 0:
        Printer.red('search_and_fuse - no matched points')
        return replace_points
    
    # extract from sim3 Scw=[s*Rcw, tcw; 0, 1] the corresponding se3 transformation Tcw=[Rcw, tcw/s]
    if isinstance(Scw, np.ndarray):
        sRcw = Scw[:3,:3]
        scw = math.sqrt(np.dot(sRcw[0,:3], sRcw[0,:3]))
        Rcw = sRcw / scw
        tcw = Scw[:3,3]/scw
    elif isinstance(Scw, Sim3Pose):
        scw = Scw.s
        Rcw = Scw.R
        tcw = Scw.t/ scw
    else: 
        raise TypeError("Unsupported type '{}' for Scw".format(type(Scw)))
    
    # check if points are visible     
    good_pts_visible, good_projs, good_depths, good_dists = are_map_points_visible_in_frame(good_pts, keyframe, Rcw, tcw)
        
    if np.sum(good_pts_visible) == 0:
        Printer.red('search_and_fuse - no visible points')
        return replace_points
    
    predicted_levels = predict_detection_levels(good_pts, good_dists) 
    kp_scale_factors = FrameShared.feature_manager.scale_factors[predicted_levels]              
    radiuses = max_reproj_distance * kp_scale_factors     
    
    kd_idxs = keyframe.kd.query_ball_point(good_projs[:,:2], radiuses)    

    for idx,p,j in zip(good_pts_idxs,good_pts,range(len(good_pts))):            
                
        if not good_pts_visible[j] or p.is_bad:     # point not visible in frame or point is bad 
            #print('p[%d] visible: %d, bad: %d' % (i, int(good_pts_visible[j]), int(p.is_bad))) 
            continue  
                  
        if p.is_in_keyframe(keyframe):    # we already matched this map point to this keyframe
            #print('p[%d] already in keyframe' % (i)) 
            continue
                     
        predicted_level = predicted_levels[j]
        
        best_dist = math.inf      
        best_kd_idx = -1        
            
        # find closest keypoints of frame        
        proj = good_projs[j]

        kd_idxs_j = kd_idxs[j]                
            
        inv_level_sigmas2 = FrameShared.feature_manager.inv_level_sigmas2
                        
        for h, kd_idx in enumerate(kd_idxs_j):             
                
            # check detection level     
            kp_level = keyframe.octaves[kd_idx]    
            if (kp_level<predicted_level-1) or (kp_level>predicted_level):   
                #print('p[%d] wrong predicted level **********************************' % (i))                       
                continue
                                    
            descriptor_dist = p.min_des_distance(keyframe.des[kd_idx])
            #print('p[%d] descriptor_dist %f **********************************' % (i,descriptor_dist))            
            
            if descriptor_dist < best_dist:                                      
                best_dist = descriptor_dist
                best_kd_idx = kd_idx   
                              
                                                            
        if best_dist < max_descriptor_distance:                      
            p_keyframe = keyframe.get_point_match(best_kd_idx)
            # if there is already a map point replace it 
            if p_keyframe is not None:
                if not p_keyframe.is_bad:
                    replace_points[idx] = p_keyframe
            else:
                p.add_observation(keyframe, best_kd_idx) 
                #p.update_info()    # done outside!
            fused_pts_count += 1       
                       
    return replace_points     


# search new matches between unmatched map points of kf1 and kf2 by using a know sim3 transformation (guided matching)
# in:
#   kf1, kf2: keyframes
#   idxs1, idxs2:  kf1.points(idxs1[i]) is matched with kf2.points(idxs2[i])  
#   s12, R12, t12: sim3 transformation that guides the matching
# out: 
#   new_matches12: where kf2.points(new_matches12[i]) is matched to i-th map point in kf1 (includes the input matches) if new_matches12[i]>0
#   new_matches21: where kf1.points(new_matches21[i]) is matched to i-th map point in kf2 (includes the input matches) if new_matches21[i]>0
def search_by_sim3(kf1: KeyFrame, kf2: KeyFrame, 
                   idxs1, idxs2, 
                   s12, R12, t12, 
                   max_reproj_distance=Parameters.kMaxReprojectionDistanceSim3, 
                   max_descriptor_distance=None,
                   print_fun=None):
    if max_descriptor_distance is None:
        max_descriptor_distance = Parameters.kMaxDescriptorDistance
        
    assert(len(idxs1) == len(idxs2))        
    # Sim3 transformations between cameras
    sR12 = s12 * R12
    sR21 = (1.0 / s12) * R12.T
    t21 = -sR21 @ t12

    map_points1 = kf1.get_points() # get all map points of kf1
    n1 = len(map_points1)
    new_matches12 = np.full(n1, -1, dtype=np.int32) # kf2.points(new_matches12[i]) is matched to i-th map point in kf1 if new_matches12[i]>0 (from 1 to 2)
    good_points1 = np.array([True if mp is not None and not mp.is_bad else False for mp in map_points1])
    
    map_points2 = kf2.get_points() # get all map points of kf2
    n2 = len(map_points2)
    new_matches21 = np.full(n2, -1, dtype=np.int32) # kf1.points(new_matches21[i]) is matched to i-th map point in kf2 if new_matches21[i]>0 (from 2 to 1)
    good_points2 = np.array([True if mp is not None and not mp.is_bad else False for mp in map_points2])
        
    for idx1, idx2 in zip(idxs1, idxs2):
        # Integrate the matches we already have as input into the output
        if good_points1[idx1] and good_points2[idx2]:
            new_matches12[idx1] = idx2
            new_matches21[idx2] = idx1        
        
    # if print_fun is not None: 
    #     print_fun(f'search_by_sim3: starting num mp matches: {np.sum(new_matches12!=-1)}')
        
    # Find unmatched map points
    unmatched_idxs1 = [idx for idx in range(n1) if good_points1[idx] and new_matches12[idx]<0]
    unmatched_map_points1 = map_points1[unmatched_idxs1]
    unmatched_idxs2 = [idx for idx in range(n2) if good_points2[idx] and new_matches21[idx]<0]
    unmatched_map_points2 = map_points2[unmatched_idxs2]
        
    # if print_fun is not None:
    #     print_fun(f'search_by_sim3: found: {len(unmatched_idxs1)} unmatched map points of kf1 {kf1.id}, {len(unmatched_idxs2)} unmatched map points of kf2 {kf2.id}')

    scale_factors = FrameShared.feature_manager.scale_factors
    
    # check which unmatched points of kf1 are visible on kf2 
    visible_flags_21, projs_21, depths_21, dists_21 = \
        are_map_points_visible(kf1, kf2, unmatched_map_points1, sR21, t21)
        
    num_visible_21 = np.sum(visible_flags_21)
    # if print_fun is not None:    
    #     print_fun(f'search_by_sim3: {num_visible_21} map points of kf1 {kf1.id} are visible on kf2 {kf2.id}')
                
    if num_visible_21 > 0: 
        predicted_levels = predict_detection_levels(unmatched_map_points1, dists_21) 
        kp_scale_factors = scale_factors[predicted_levels]              
        radiuses = max_reproj_distance * kp_scale_factors     
        kd2_idxs = kf2.kd.query_ball_point(projs_21[:,:2], radiuses)   # search NN kps on kf2
        
        for i1, mp1 in enumerate(unmatched_map_points1):
            kd2_idxs_i = kd2_idxs[i1]                  
            predicted_level = predicted_levels[i1]                
                            
            best_dist = float('inf')
            best_idx = -1                                      
            for kd2_idx in kd2_idxs_i:             
                # check detection level     
                kp_level = kf2.octaves[kd2_idx]    
                if (kp_level<predicted_level-1) or (kp_level>predicted_level):                       
                    continue                      
                
                dist = mp1.min_des_distance(kf2.des[kd2_idx])

                if dist < best_dist:
                    best_dist = dist
                    best_idx = kd2_idx

            if best_dist <= max_descriptor_distance:
                if new_matches21[best_idx]==-1:
                    new_matches12[unmatched_idxs1[i1]] = best_idx
                 
    # check which unmatched points of kf2 are visible on kf1 
    visible_flags_12, projs_12, depths_12, dists_12 = \
        are_map_points_visible(kf2, kf1, unmatched_map_points2, sR12, t12)
        
    num_visible_12 = np.sum(visible_flags_12)
    # if print_fun is not None:         
    #     print_fun(f'search_by_sim3: {num_visible_12} map points of kf2 {kf2.id} are visible on kf1 {kf1.id}')        
          
    if num_visible_12 > 0:
        predicted_levels = predict_detection_levels(unmatched_map_points2, dists_12) 
        kp_scale_factors = scale_factors[predicted_levels]              
        radiuses = max_reproj_distance * kp_scale_factors     
        kd1_idxs = kf1.kd.query_ball_point(projs_12[:,:2], radiuses)   # search NN kps on kf1
        
        for i2, mp2 in enumerate(unmatched_map_points2):
            kd1_idxs_i = kd1_idxs[i2]                  
            predicted_level = predicted_levels[i2]                
                            
            best_dist = float('inf')
            best_idx = -1                                      
            for kd1_idx in kd1_idxs_i:             
                # check detection level     
                kp_level = kf1.octaves[kd1_idx]    
                if (kp_level<predicted_level-1) or (kp_level>predicted_level):                       
                    continue                      
                
                dist = mp2.min_des_distance(kf1.des[kd1_idx])

                if dist < best_dist:
                    best_dist = dist
                    best_idx = kd1_idx

            if best_dist <= max_descriptor_distance:
                if new_matches12[best_idx]==-1:
                    new_matches21[unmatched_idxs2[i2]] = best_idx

    # if print_fun is not None:    
    #     print_fun(f'search_by_sim3: new matches before check: 1->2: {np.sum(new_matches12!=-1)}, 2->1: {np.sum(new_matches21!=-1)}')

    # Check agreement
    num_matches_found = 0
    for i1 in range(n1):
        idx2 = new_matches12[i1] # index of kf2 point that matches with i1-th kf1 point
        if idx2 >= 0:
            idx1 = new_matches21[idx2] # index of kf1 point that matches with idx2-th kf2 point
            if idx1 != i1: # reset if mismatch
                new_matches12[i1] = -1
                new_matches21[idx2] = -1
            else: 
                num_matches_found += 1
 
    # if print_fun is not None:    
    #     print_fun(f'search_by_sim3: num matches found after final check: {num_matches_found}')
    #     print_fun(f'search_by_sim3: new matches after check: 1->2: {np.sum(new_matches12!=-1)}, 2->1: {np.sum(new_matches21!=-1)}')
        
    return num_matches_found, new_matches12, new_matches21



import numpy as np
import cv2
import time
from scipy.spatial import KDTree
from typing import Dict, Tuple, List, Optional, Any

# --- IMPORT YOUR FRAME CLASS ---
# Make sure this class definition is available in the scope
# from your_module import Frame, FrameBase # Or however you import it

# Assume MapSnapshot is a dictionary structure like:
# snapshot = {
#     'timestamp': float,
#     'map_points': {'points': np.array, 'colors': np.array, 'descriptors': np.array},
#     'local_map_points': {'points': np.array, 'colors': np.array, 'descriptors': np.array},
#     # ... other data
# }

# Optional Open3D import for visualization
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("Warning: Open3D not available. 3D visualization will be skipped.")
    print("Install with 'pip install open3d'")

# --- Helper Functions ---

def hamming_distance(des1: np.ndarray, des2: np.ndarray) -> int:
    """Calculates Hamming distance between two ORB descriptors."""
    # Ensure descriptors are uint8
    if des1.dtype != np.uint8:
        des1 = np.uint8(des1)
    if des2.dtype != np.uint8:
        des2 = np.uint8(des2)
    # Handle potential size mismatch if descriptors are somehow invalid
    if des1.shape != des2.shape:
        # print(f"Warning: Descriptor shape mismatch: {des1.shape} vs {des2.shape}")
        return 9999 # Return a large distance
    try:
        return int(cv2.norm(des1, des2, cv2.NORM_HAMMING))
    except cv2.error as e:
        # print(f"Warning: cv2.norm error calculating Hamming distance: {e}")
        return 9999 # Return a large distance


def find_best_descriptor_match(
    target_des: np.ndarray,
    candidate_indices: List[int],
    candidate_descriptors: np.ndarray,
    max_distance: int,
    ratio_test: float
) -> Tuple[Optional[int], Optional[int]]:
    """
    Finds the best descriptor match among candidates using Hamming distance
    and Lowe's ratio test.

    Returns:
        Tuple[Optional[int], Optional[int]]: (best_match_idx, best_distance) or (None, None)
    """
    best_match_idx = None
    best_dist = float('inf')
    second_best_dist = float('inf')

    if not candidate_indices:
        return None, None

    # Pre-check target descriptor
    if target_des is None:
        # print("Warning: Target descriptor is None.")
        return None, None

    for idx in candidate_indices:
        # Check index bounds and if candidate descriptor exists
        if idx >= len(candidate_descriptors) or candidate_descriptors[idx] is None:
            # print(f"Warning: Invalid candidate index {idx} or descriptor is None.")
            continue

        dist = hamming_distance(target_des, candidate_descriptors[idx])

        if dist < best_dist:
            second_best_dist = best_dist
            best_dist = dist
            best_match_idx = idx
        elif dist < second_best_dist:
            second_best_dist = dist

    if best_match_idx is not None and best_dist <= max_distance:
        # Apply ratio test
        if second_best_dist == float('inf') or best_dist < ratio_test * second_best_dist:
            return best_match_idx, int(best_dist) # Return int distance

    return None, None


# --- Function 1: Search Common Points Between Snapshots ---

def search_common_points_between_snapshots(
    snapshot_prev: Dict[str, Any],
    snapshot_cur: Dict[str, Any],
    use_local_map: bool = False,
    max_3d_distance: float = 0.2, # Max distance in 3D space (meters)
    max_descriptor_distance: int = 50, # Max Hamming distance for ORB
    ratio_test: float = 0.8, # Lowe's ratio test threshold
    visualize: bool = False
) -> Tuple[List[int], List[int], List[np.ndarray], List[np.ndarray]]:
    """
    Finds common points between two map snapshots based on 3D proximity
    and ORB descriptor similarity.

    Args:
        snapshot_prev: Dictionary representing the previous map snapshot.
        snapshot_cur: Dictionary representing the current map snapshot.
        use_local_map: If True, uses 'local_map_points', otherwise 'map_points'.
        max_3d_distance: Maximum Euclidean distance in 3D for points to be considered neighbors.
        max_descriptor_distance: Maximum Hamming distance for descriptors to match.
        ratio_test: Threshold for Lowe's ratio test on descriptor distances.
        visualize: If True, attempts to visualize the matches in 3D using Open3D.

    Returns:
        Tuple containing:
        - List[int]: Indices of matched points in snapshot_prev.
        - List[int]: Indices of matched points in snapshot_cur.
        - List[np.ndarray]: 3D coordinates of matched points from snapshot_prev.
        - List[np.ndarray]: 3D coordinates of matched points from snapshot_cur.
    """
    print(f"Searching common points between snapshots at {snapshot_prev.get('timestamp', 'N/A')} "
          f"and {snapshot_cur.get('timestamp', 'N/A')}")
    start_time = time.time()

    map_key = 'local_map_points' if use_local_map else 'map_points'

    # Safely get data, handling potential missing keys or None values
    data_prev = snapshot_prev.get(map_key, {})
    points_prev = data_prev.get('points')
    des_prev = data_prev.get('descriptors')
    colors_prev = data_prev.get('colors') # For visualization

    data_cur = snapshot_cur.get(map_key, {})
    points_cur = data_cur.get('points')
    des_cur = data_cur.get('descriptors')
    colors_cur = data_cur.get('colors') # For visualization

    # --- Input Validation ---
    if points_prev is None or des_prev is None or \
       points_cur is None or des_cur is None or \
       len(points_prev) == 0 or len(points_cur) == 0:
        print("Warning: Missing points or descriptors in one or both snapshots, or snapshots empty.")
        return [], [], [], []

    if len(points_prev) != len(des_prev):
         print(f"Warning: Mismatch between point count ({len(points_prev)}) and "
               f"descriptor count ({len(des_prev)}) in snapshot_prev. Skipping.")
         return [], [], [], []
    if len(points_cur) != len(des_cur):
         print(f"Warning: Mismatch between point count ({len(points_cur)}) and "
               f"descriptor count ({len(des_cur)}) in snapshot_cur. Skipping.")
         return [], [], [], []

    print(f"  Prev snapshot: {len(points_prev)} points.")
    print(f"  Cur snapshot: {len(points_cur)} points.")

    # --- KD-Tree ---
    try:
        print("  Building KD-Tree for current snapshot points...")
        kdtree_cur = KDTree(points_cur)
        print("  KD-Tree built.")
    except Exception as e:
        print(f"Error building KD-Tree for current points: {e}")
        return [], [], [], []

    # --- Matching Loop ---
    matched_indices_prev = []
    matched_indices_cur = []
    matched_points_prev_list = [] # Use list temporarily
    matched_points_cur_list = [] # Use list temporarily

    print("  Searching for matches...")
    for i, p_prev in enumerate(points_prev):
        # Check if previous descriptor exists
        if des_prev[i] is None:
            continue

        # Find points in the current snapshot within the 3D distance threshold
        try:
            candidate_indices_cur = kdtree_cur.query_ball_point(p_prev, max_3d_distance)
        except ValueError as e:
            # Can happen if p_prev has incorrect dimension or NaN/inf
            # print(f"Warning: Skipping point {i} due to KD-Tree query error: {e}")
            continue
        except Exception as e:
             print(f"Error querying KD-Tree at index {i}: {e}")
             continue # Skip this point

        if not candidate_indices_cur:
            continue

        # Find the best descriptor match among the 3D neighbors
        best_match_idx_cur, best_dist = find_best_descriptor_match(
            des_prev[i],
            candidate_indices_cur,
            des_cur,
            max_descriptor_distance,
            ratio_test
        )

        if best_match_idx_cur is not None:
            # Found a match
            matched_indices_prev.append(i)
            matched_indices_cur.append(best_match_idx_cur)
            matched_points_prev_list.append(p_prev)
            matched_points_cur_list.append(points_cur[best_match_idx_cur])

    end_time = time.time()
    print(f"  Found {len(matched_indices_prev)} common points in {end_time - start_time:.3f} seconds.")

    # Convert lists to numpy arrays for return type consistency
    matched_points_prev_arr = np.array(matched_points_prev_list) if matched_points_prev_list else np.empty((0, 3))
    matched_points_cur_arr = np.array(matched_points_cur_list) if matched_points_cur_list else np.empty((0, 3))


    # --- Visualization ---
    if visualize and OPEN3D_AVAILABLE and len(matched_indices_prev) > 0:
        print("  Visualizing matches...")
        # (Visualization code remains the same as previous version)
        try:
            pcd_prev = o3d.geometry.PointCloud()
            pcd_prev.points = o3d.utility.Vector3dVector(matched_points_prev_arr)
            if colors_prev is not None and len(colors_prev) == len(points_prev):
                 matched_colors_prev = colors_prev[matched_indices_prev]
                 if matched_colors_prev.dtype != np.float64 and matched_colors_prev.dtype != np.float32:
                      matched_colors_prev = matched_colors_prev.astype(np.float32) / 255.0
                 pcd_prev.colors = o3d.utility.Vector3dVector(matched_colors_prev % 1.0) # Ensure in [0,1]
            else:
                 pcd_prev.paint_uniform_color([1, 0, 0]) # Red

            pcd_cur = o3d.geometry.PointCloud()
            pcd_cur.points = o3d.utility.Vector3dVector(matched_points_cur_arr)
            if colors_cur is not None and len(colors_cur) == len(points_cur):
                 matched_colors_cur = colors_cur[matched_indices_cur]
                 if matched_colors_cur.dtype != np.float64 and matched_colors_cur.dtype != np.float32:
                      matched_colors_cur = matched_colors_cur.astype(np.float32) / 255.0
                 pcd_cur.colors = o3d.utility.Vector3dVector(matched_colors_cur % 1.0) # Ensure in [0,1]
            else:
                 pcd_cur.paint_uniform_color([0, 0, 1]) # Blue

            lines = [[i, i + len(matched_points_prev_arr)] for i in range(len(matched_points_prev_arr))]
            line_colors = [[0, 1, 0] for _ in range(len(lines))]
            if pcd_prev.has_colors() and pcd_cur.has_colors():
                 try:
                      cols_p = np.asarray(pcd_prev.colors)
                      cols_c = np.asarray(pcd_cur.colors)
                      line_colors = (cols_p + cols_c) / 2.0
                 except Exception:
                      line_colors = [[0, 1, 0] for _ in range(len(lines))]

            line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(np.vstack((matched_points_prev_arr, matched_points_cur_arr))),
                lines=o3d.utility.Vector2iVector(lines)
            )
            line_set.colors = o3d.utility.Vector3dVector(line_colors)

            print("  Displaying Open3D window...")
            o3d.visualization.draw_geometries([pcd_prev, pcd_cur, line_set],
                                               window_name=f"Snapshot Matches {snapshot_prev.get('timestamp', 'Prev')} <-> {snapshot_cur.get('timestamp', 'Cur')}")
            print("  Open3D window closed.")
        except Exception as e:
            print(f"Error during Open3D visualization: {e}")


    return matched_indices_prev, matched_indices_cur, matched_points_prev_arr, matched_points_cur_arr

# --- Function 2: Search Common Points Between Snapshot and Frame ---

def search_common_points_snapshot_frame(
    snapshot: Dict[str, Any],
    frame: Frame, # Use the actual Frame type hint
    use_local_map: bool = False,
    max_reproj_distance: float = 5.0, # Max distance in pixels for reprojection match
    max_descriptor_distance: int = 50, # Max Hamming distance for ORB
    ratio_test: float = 0.8, # Lowe's ratio test threshold
    visualize: bool = False,
    frame_img: Optional[np.ndarray] = None # Optional image for visualization
) -> Tuple[List[int], List[int], List[np.ndarray], List[np.ndarray]]:
    """
    Finds common points between a map snapshot (3D points) and a frame (2D keypoints)
    using projection and descriptor matching.

    Args:
        snapshot: Dictionary representing the map snapshot.
        frame: The Frame object containing keypoints, descriptors, pose, camera.
        use_local_map: If True, uses 'local_map_points', otherwise 'map_points'.
        max_reproj_distance: Max pixel distance for a keypoint to match a projection.
        max_descriptor_distance: Maximum Hamming distance for descriptors to match.
        ratio_test: Threshold for Lowe's ratio test on descriptor distances.
        visualize: If True, attempts to visualize the matches on the frame image using OpenCV.
        frame_img: Optional image corresponding to the frame object for visualization.

    Returns:
        Tuple containing:
        - List[int]: Indices of matched points in the snapshot.
        - List[int]: Indices of matched keypoints in the frame.
        - List[np.ndarray]: 3D coordinates of matched points from the snapshot.
        - List[np.ndarray]: 2D coordinates of matched keypoints from the frame.
    """
    print(f"Searching common points between snapshot at {snapshot.get('timestamp', 'N/A')} "
          f"and frame {getattr(frame, 'id', 'N/A')}")
    start_time = time.time()

    map_key = 'local_map_points' if use_local_map else 'map_points'

    # --- Safely get snapshot data ---
    data_snap = snapshot.get(map_key, {})
    snap_points_3d = data_snap.get('points')
    snap_des = data_snap.get('descriptors')

    # --- Snapshot Input Validation ---
    if snap_points_3d is None or snap_des is None or len(snap_points_3d) == 0:
        print("Warning: Missing points or descriptors in snapshot, or snapshot empty.")
        return [], [], [], []
    if len(snap_points_3d) != len(snap_des):
        print(f"Warning: Mismatch between point count ({len(snap_points_3d)}) and "
              f"descriptor count ({len(snap_des)}) in snapshot. Skipping.")
        return [], [], [], []

    # --- Safely get frame data ---
    try:
        # Access attributes using getattr for safety in case Frame structure varies slightly
        frame_kps_2d = getattr(frame, 'kpsu', None)
        frame_des = getattr(frame, 'des', None)
        # Access the KD-Tree property
        frame_kd_tree = getattr(frame, 'kd', None)
        # Check if pose and camera attributes exist
        if not hasattr(frame, 'pose') or not hasattr(frame, 'camera'):
             raise AttributeError("Frame object missing 'pose' or 'camera' attribute.")
        if not hasattr(frame, 'project_points'):
             raise AttributeError("Frame object missing 'project_points' method.")
        if not hasattr(frame, 'are_in_image'):
             raise AttributeError("Frame object missing 'are_in_image' method.")

        if frame_kps_2d is None or frame_des is None or frame_kd_tree is None:
             raise ValueError("Frame is missing keypoints (kpsu), descriptors (des), or KD-Tree (kd).")
        if len(frame_kps_2d) != len(frame_des):
             raise ValueError("Frame keypoint count and descriptor count mismatch.")
        if len(frame_kps_2d) == 0:
            print("Warning: Frame has no keypoints.")
            return [], [], [], []

    except (AttributeError, ValueError) as e:
        print(f"Error accessing Frame attributes or invalid Frame data: {e}")
        return [], [], [], []

    print(f"  Snapshot: {len(snap_points_3d)} points.")
    print(f"  Frame: {len(frame_kps_2d)} keypoints.")

    # --- Projection ---
    print("  Projecting snapshot points onto frame...")
    try:
        # project_points is expected to handle internal pose/camera details
        projections_2d, depths = frame.project_points(snap_points_3d)
        # Check visibility using the frame's method
        visible_mask = frame.are_in_image(projections_2d, depths)
        print(f"  {np.sum(visible_mask)} snapshot points are potentially visible.")

    except Exception as e:
        print(f"Error during projection or visibility check using frame methods: {e}")
        return [], [], [], []

    # --- Matching Loop ---
    matched_indices_snap = []
    matched_indices_frame = []
    matched_points_snap_3d_list = []
    matched_kps_frame_2d_list = []

    print("  Searching for matches...")
    visible_indices = np.where(visible_mask)[0]
    for i in visible_indices:
        p_snap_3d = snap_points_3d[i]
        des_snap = snap_des[i]
        proj_2d = projections_2d[i][:2] # Use only (u, v) coordinates

        # Check if snapshot descriptor exists
        if des_snap is None:
            continue

        # Find candidate keypoints in the frame within the reprojection distance
        try:
            candidate_indices_frame = frame_kd_tree.query_ball_point(
                proj_2d,
                max_reproj_distance
            )
        except ValueError as e:
            # print(f"Warning: Skipping snapshot point {i} due to Frame KD-Tree query error: {e}")
            continue
        except Exception as e:
            print(f"Error querying frame KD-Tree for snapshot point {i}: {e}")
            continue # Skip this snapshot point

        if not candidate_indices_frame:
            continue

        # Find the best descriptor match among the frame keypoint candidates
        best_match_idx_frame, best_dist = find_best_descriptor_match(
            des_snap,
            candidate_indices_frame,
            frame_des,
            max_descriptor_distance,
            ratio_test
        )

        if best_match_idx_frame is not None:
            # Found a match
            matched_indices_snap.append(i)
            matched_indices_frame.append(best_match_idx_frame)
            matched_points_snap_3d_list.append(p_snap_3d)
            matched_kps_frame_2d_list.append(frame_kps_2d[best_match_idx_frame])

    end_time = time.time()
    print(f"  Found {len(matched_indices_snap)} snapshot-frame matches in {end_time - start_time:.3f} seconds.")

    # Convert lists to numpy arrays
    matched_points_snap_3d_arr = np.array(matched_points_snap_3d_list) if matched_points_snap_3d_list else np.empty((0, 3))
    matched_kps_frame_2d_arr = np.array(matched_kps_frame_2d_list) if matched_kps_frame_2d_list else np.empty((0, 2))

    # --- Visualization ---
    if visualize and frame_img is not None and len(matched_indices_frame) > 0:
        print("  Visualizing matches...")
        # (Visualization code remains the same as previous version)
        try:
            vis_img = frame_img.copy()
            if len(vis_img.shape) == 2: # Grayscale
                vis_img = cv2.cvtColor(vis_img, cv2.COLOR_GRAY_BGR)

            # Draw all keypoints in the frame
            for kp_idx in range(len(frame_kps_2d)):
                kp = frame_kps_2d[kp_idx]
                # Check if kp is None before drawing
                if kp is not None:
                    cv2.circle(vis_img, tuple(kp.astype(int)), 2, (0, 255, 0), -1) # Green circles


            # Draw matched keypoints and projections
            for idx_frame, idx_snap in zip(matched_indices_frame, matched_indices_snap):
                 kp = frame_kps_2d[idx_frame]
                 # Ensure kp is not None before proceeding
                 if kp is None:
                     continue

                 # Get projection location for this match
                 proj_match = projections_2d[idx_snap][:2].astype(int)

                 # Draw projection location (optional)
                 cv2.circle(vis_img, tuple(proj_match), 4, (255, 255, 0), 1) # Cyan circle for projection

                 # Draw matched keypoint
                 cv2.circle(vis_img, tuple(kp.astype(int)), 5, (0, 0, 255), 1) # Red circle for match

                 # Draw line from projection to match
                 cv2.line(vis_img, tuple(proj_match), tuple(kp.astype(int)), (255, 0, 255), 1) # Magenta line

            print("  Displaying OpenCV window...")
            cv2.imshow(f"Snapshot-Frame Matches (Snap: {snapshot.get('timestamp', 'N/A')}, Frame: {getattr(frame, 'id', 'N/A')})", vis_img)
            print("  Press any key in the OpenCV window to continue...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            print("  OpenCV window closed.")
        except Exception as e:
            print(f"Error during OpenCV visualization: {e}")
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    return matched_indices_snap, matched_indices_frame, matched_points_snap_3d_arr, matched_kps_frame_2d_arr


#