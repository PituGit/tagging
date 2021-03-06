# -*- coding: utf-8 -*-
"""

@author: Oscar, Esteban, Maxi
"""
import re
import numpy as np
import ColorNaming as cn
from skimage import color
import KMeans as km

def NIUs():
    return 1458082, 1455249, 1455189

def loadGT(fileName):
    """@brief   Loads the file with groundtruth content
    
    @param  fileName  STRING    name of the file with groundtruth
    
    @return groundTruth LIST    list of tuples of ground truth data
                                (Name, [list-of-labels])
    """

    groundTruth = []
    fd = open(fileName, 'r')
    for line in fd:
        splitLine = line.split(' ')[:-1]
        labels = [''.join(sorted(filter(None,re.split('([A-Z][^A-Z]*)',l)))) for l in splitLine[1:]]
        groundTruth.append( (splitLine[0], labels) )
        
    return groundTruth


def evaluate(description, GT, options):
    """@brief   EVALUATION FUNCTION
    @param description LIST of color name lists: contain one lsit of color labels for every images tested
    @param GT LIST images to test and the real color names (see  loadGT)
    @options DICT  contains options to control metric, ...
    @return mean_score,scores mean_score FLOAT is the mean of the scores of each image
                              scores     LIST contain the similiraty between the ground truth list of color names and the obtained
    """
#########################################################
##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
##  AND CHANGE FOR YOUR OWN CODE
#########################################################
    score = 0.0
    scoreList = np.zeros((len(description)))
    for i, desc in enumerate(description):
        scoreList[i] = similarityMetric(desc, GT[i][1], options)
    score = scoreList.mean()
    return score, list(scoreList)



def similarityMetric(Est, GT, options):
    """@brief   SIMILARITY METRIC
    @param Est LIST  list of color names estimated from the image ['red','green',..]
    @param GT LIST list of color names from the ground truth
    @param options DICT  contains options to control metric, ...
    @return S float similarity between label LISTs
    """
    
    if options == None:
        options = {}
    if not 'metric' in options:
        options['metric'] = 'basic'
        
#########################################################
##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
##  AND CHANGE FOR YOUR OWN CODE
#########################################################
    if options['metric'].lower() == 'basic'.lower():
        intersection = set(Est).intersection(set(GT))
        return len(intersection) / float(len(Est))     
    else:
        return 0
        
def getLabels(kmeans, options):
    """@brief   Labels all centroids of kmeans object to their color names
    
    @param  kmeans  KMeans      object of the class KMeans
    @param  options DICTIONARY  options necessary for labeling
    
    @return colors  LIST    colors labels of centroids of kmeans object
    @return ind     LIST    indexes of centroids with the same color label
    """

#########################################################
##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
##  AND CHANGE FOR YOUR OWN CODE
#########################################################
##  remind to create composed labels if the probability of 
##  the best color label is less than  options['single_thr']
    name_colors = cn.colors
    colors = []
    ind = {}

    for centroid in kmeans.centroids:
        colorIndex = np.flip(np.argsort(centroid), 0) #Orden descendente
        sortedCentroid = np.flip(np.sort(centroid), 0)  #Orden descendente
        
        if sortedCentroid[0] < options['single_thr']:
            composed = sorted([name_colors[colorIndex[0]], name_colors[colorIndex[1]]])
            colors.append(composed[0] + composed[1])
        else:
            colors.append(name_colors[colorIndex[0]])

    for i, color in enumerate(colors):
        ind.setdefault(color, []).append(i)

    ind = [ind[color] for color in sorted(ind.keys())]
    colors = sorted(list(set(colors)))

    return colors, ind;

def processImage(im, options):
    """@brief   Finds the colors present on the input image
    
    @param  im      LIST    input image
    @param  options DICTIONARY  dictionary with options
    
    @return colors  LIST    colors of centroids of kmeans object
    @return indexes LIST    indexes of centroids with the same label
    @return kmeans  KMeans  object of the class KMeans
    """

#########################################################
##  YOU MUST ADAPT THE CODE IN THIS FUNCTIONS TO:
##  1- CHANGE THE IMAGE TO THE CORRESPONDING COLOR SPACE FOR KMEANS
##  2- APPLY KMEANS ACCORDING TO 'OPTIONS' PARAMETER
##  3- GET THE NAME LABELS DETECTED ON THE 11 DIMENSIONAL SPACE
#########################################################

##  1- CHANGE THE IMAGE TO THE CORRESPONDING COLOR SPACE FOR KMEANS
    if options['colorspace'].lower() == 'ColorNaming'.lower():  
        im = cn.ImColorNamingTSELabDescriptor(im)
    elif options['colorspace'].lower() == 'RGB'.lower():        
        pass  #Ya estamos en RGB
    elif options['colorspace'].lower() == 'Lab'.lower():        
        im = color.rgb2lab(im)
        im = np.array(im).reshape((-1,3))


##  2- APPLY KMEANS ACCORDING TO 'OPTIONS' PARAMETER
    if options['K']<2: # find the bes K
        kmeans = km.KMeans(im, 0, options)
        kmeans.bestK() #bestK segons fitting
    else:
        kmeans = km.KMeans(im, options['K'], options) 
        kmeans.run()

##  3- GET THE NAME LABELS DETECTED ON THE 11 DIMENSIONAL SPACE
    if options['colorspace'].lower() == 'RGB'.lower():        
        kmeans.centroids = cn.ImColorNamingTSELabDescriptor(kmeans.centroids)
    elif options['colorspace'].lower() == 'Lab'.lower():
        pass
        

#########################################################
##  THE FOLLOWING 2 END LINES SHOULD BE KEPT UNMODIFIED
#########################################################
    colors, which = getLabels(kmeans, options)   
    return colors, which, kmeans
