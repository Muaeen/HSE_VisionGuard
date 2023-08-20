# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from simple_image_download import simple_image_download as simp

re = simp.simple_image_download

key_w = ["building workers","construction workers"]

for kw in key_w:
    re().download(kw, 150)
    
    