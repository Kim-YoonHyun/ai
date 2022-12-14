from PIL import Image
import os
import numpy as np 
import time

for i in range(1000):
    s = 5*(0.99 ** i)
    print(i, s)
    
    time.sleep(s)
    