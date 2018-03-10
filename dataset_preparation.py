#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 15:34:31 2018

@author: zpr
"""

import csv
import cv2
import numpy as np


lines = []



with open('/media/zpr/5aa7062e-a1a2-4b29-85cb-5756318d57ee/Udacity/CarND-Behavioral-Cloning-P3/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        print(line)