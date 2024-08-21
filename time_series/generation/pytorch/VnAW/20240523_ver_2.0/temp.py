import sys
import os
import numpy as np

def main():
    ground = []
    for i in range(3):
        ground.append([])
        for j in range(3):
            if j == 0:
                ground[i].append([i+1])
            else:
                ground[i].append([])
                cover = 1
                
if __name__ == '__main__':
    main()