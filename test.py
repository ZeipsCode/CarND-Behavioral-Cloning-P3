#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 10:31:44 2018

@author: zpr
"""

import numpy as np

a = np.arange(20)

print(a)

a = np.clip(a, 1,9)

print(a)