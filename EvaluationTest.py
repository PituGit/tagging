# -*- coding: utf-8 -*-
"""

@author: ramon
"""
from skimage import io
import matplotlib.pyplot as plt
import os
import time 

if os.path.isfile('TeachersLabels.py') and True: 
    import TeachersLabels as lb
else:
    import Labels as lb



plt.close("all")
if __name__ == "__main__":

    #'colorspace': 'RGB', 'Lab' o 'ColorNaming'
    possible_k = [3, 5, 7]
    possible_thr = [0.2, 0.6, 0.8]
    possible_init = ["first", "random"]

    ImageFolder = 'Images'
    GTFile = 'LABELSsmall.txt'
    
    GTFile = ImageFolder + '/' + GTFile
    GT = lb.loadGT(GTFile)

    for k in possible_k:
        for thr in possible_thr:
            for init in possible_init:
                options = {'colorspace':'RGB', 'K':k, 'synonyms':False, 'single_thr':thr, 'verbose':False, 'km_init':init, 'metric':'basic'}
                t = time.time()
                DBcolors = []
                print("|[", end = '', flush=True)
                for gt in GT:
                    print("//", end = '', flush=True)
                    im = io.imread(ImageFolder+"/"+gt[0])    
                    colors,_,_ = lb.processImage(im, options)
                    DBcolors.append(colors)
                print("]|")
                
                encert, _ = lb.evaluate(DBcolors, GT, options)
                print(str(k) + ", " + str(thr) + ", " + init)
                print("> Encert promig: "+ '%.2f' % (encert*100) + '%')
                print("> " + str(time.time() - t))