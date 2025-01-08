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
''' 
Explanation:

    Purpose: This code defines various scoring functions used to compare global image descriptors in loop closure detection. It provides implementations for SAD (Sum of Absolute Differences) and cosine similarity, including a version for PyTorch tensors.
    SCoreType (Enum):
        This enum defines the different types of scoring functions available.
    ScoreBase (Abstract Class):
        This is an abstract base class for all scoring functions.
        It defines the common interface, including the __call__ method that allows the object to be called as a function.
    ScoreSad:
        This class implements the SAD scoring function.
        It computes the average SAD between two descriptors, handling potential NaN (Not a Number) values.
    ScoreCosine:
        This class implements the cosine similarity scoring function.
        It computes the cosine similarity between two descriptors, which measures the angle between them.
    ScoreTorchCosine:
        This class implements the cosine similarity scoring function using PyTorch tensors.
        It's useful when working with deep learning models that output PyTorch tensors.

Key Concepts:

    Loop Closure: The process of recognizing previously visited places in a map.
    Global Descriptor: A compact representation of an image that captures its overall appearance.
    Scoring Function: A function that measures the similarity between two global descriptors.
    SAD (Sum of Absolute Differences): A simple distance metric that measures the absolute difference between corresponding elements of two descriptors.
    Cosine Similarity: A similarity metric that measures the angle between two vectors.
'''

import os
import time
import math 
import numpy as np
import cv2
import sys
from enum import Enum

from utils_sys import getchar, Printer 

from typing import List

from config_parameters import Parameters
import torch

import traceback


kVerbose = True

kMinDeltaFrameForMeaningfulLoopClosure = Parameters.kMinDeltaFrameForMeaningfulLoopClosure
kMaxResultsForLoopClosure = Parameters.kMaxResultsForLoopClosure

kTimerVerbose = False

kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder
kDataFolder = kRootFolder + '/../data'


if Parameters.kLoopClosingDebugAndPrintToFile:
    from loop_detector_base import print


class SCoreType(Enum):
    COSINE = 0
    SAD = 1


# Base class
class ScoreBase:
    def __init__(self, type, worst_score, best_score):
        self.type = type
        self.worst_score = worst_score
        self.best_score = best_score
    
    # g_des1 is [1, D], g_des2 is [M, D]
    def __call__(self, g_des1, g_des2):
        pass


class ScoreSad(ScoreBase):
    def __init__(self):
        super().__init__(SCoreType.SAD, worst_score=-sys.float_info.max, best_score=0.0)
        
    @staticmethod
    def score(g_des1, g_des2):
        diff = g_des1-g_des2
        is_nan_diff = np.isnan(diff)
        nan_count_per_row = np.count_nonzero(is_nan_diff, axis=1)
        dim = diff.shape[1] - nan_count_per_row
        #print(f'dim: {dim}, diff.shape: {diff.shape}')
        diff[is_nan_diff] = 0
        return -np.sum(np.abs(diff),axis=1) / dim   # invert the sign of the standard SAD score
        
    # g_des1 is [1, D], g_des2 is [M, D]
    def __call__(self, g_des1, g_des2):
        return self.score(g_des1, g_des2)


class ScoreCosine(ScoreBase):
    def __init__(self):
        super().__init__(SCoreType.COSINE, worst_score=-1.0, best_score=1.0)
  
    @staticmethod
    def score(g_des1, g_des2):
        norm_g_des1 = np.linalg.norm(g_des1, axis=1, keepdims=True)  # g_des1 is [1, D], so norm is scalar
        norm_g_des2 = np.linalg.norm(g_des2, axis=1, keepdims=True)  # g_des2 is [M, D]
        dot_product = np.dot(g_des2, g_des1.T).ravel()
        cosine_similarity = dot_product / (norm_g_des1 * norm_g_des2.ravel())
        return cosine_similarity.ravel()
      
    # g_des1 is [1, D], g_des2 is [M, D]
    def __call__(self, g_des1, g_des2):
        return self.score(g_des1, g_des2)
  

class ScoreTorchCosine(ScoreBase):
    def __init__(self):
        super().__init__(SCoreType.COSINE, worst_score=-1.0, best_score=1.0)
  
    @staticmethod
    def score(g_des1, g_des2):
        # Ensure g_des1 is a 2D tensor of shape [1, D]
        if g_des1.dim() == 1:
            g_des1 = g_des1.unsqueeze(0)

        # Compute the norms
        norm_g_des1 = g_des1.norm(dim=1, keepdim=True)  # Shape [1, 1]
        norm_g_des2 = g_des2.norm(dim=1, keepdim=True)  # Shape [M, 1]

        # Dot product between g_des1 and each row of g_des2
        dot_product = torch.mm(g_des2, g_des1.T).squeeze()  # Shape [M]

        # Compute cosine similarity
        cosine_similarity = (dot_product / (norm_g_des1 * norm_g_des2).squeeze()).ravel()
        return cosine_similarity

    # g_des1 is [1, D], g_des2 is [M, D]
    def __call__(self, g_des1, g_des2):
        return self.score(g_des1, g_des2)
    
