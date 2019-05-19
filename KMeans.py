# -*- coding: utf-8 -*-
"""

@author: Oscar, Esteban, Maxi
"""
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d
from sklearn.decomposition import PCA
from scipy.spatial import distance_matrix


def NIUs():
    return 1458082, 1455249, 1455189
    
def distance(X,C):
    """@brief   Calculates the distance between each pixcel and each centroid 

    @param  X  numpy array PxD 1st set of data points (usually data points)
    @param  C  numpy array KxD 2nd set of data points (usually cluster centroids points)

    @return dist: PxK numpy array position ij is the distance between the 
    	i-th point of the first set an the j-th point of the second set
    """
#########################################################
##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
##  AND CHANGE FOR YOUR OWN CODE
#########################################################
    return distance_matrix(X, C) #Calcula la distancia de todos los pares posibles entre las dos matrices

class KMeans():
    
    def __init__(self, X, K, options=None):
        """@brief   Constructor of KMeans class
        
        @param  X   LIST    input data
        @param  K   INT     number of centroids
        @param  options DICT dctionary with options
        """

        self._init_X(X)                                    # LIST data coordinates
        self._init_options(options)                        # DICT options
        self._init_rest(K)                                 # Initializes de rest of the object
        
#############################################################
##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
#############################################################

        
    def _init_X(self, X):
        """@brief Initialization of all pixels
        
        @param  X   LIST    list of all pixel values. Usually it will be a numpy 
                            array containing an image NxMx3

        sets X an as an array of data in vector form (PxD  where P=N*M and D=3 in the above example)
        """
#######################################################
##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
##  AND CHANGE FOR YOUR OWN CODE
#######################################################
        self.X = (np.reshape(X, (-1, X.shape[-1])).astype("float64"))#np.reshape(X, (P,D))

            
    def _init_options(self, options):
        """@brief Initialization of options in case some fields are left undefined
        
        @param  options DICT dctionary with options

			sets de options parameters
        """
        if options == None:
            options = {}
        if not 'km_init' in options:
            options['km_init'] = 'first'
        if not 'verbose' in options:
            options['verbose'] = False
        if not 'tolerance' in options:
            options['tolerance'] = 0
        if not 'max_iter' in options:
            options['max_iter'] = np.inf
        if not 'fitting' in options:
            options['fitting'] = 'Fisher'

        self.options = options
        
#############################################################
##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
#############################################################

        
    def _init_rest(self, K):
        """@brief   Initialization of the remainig data in the class.
        
        @param  options DICT dctionary with options
        """
        self.K = K                                             # INT number of clusters
        if self.K>0:
            self._init_centroids()                             # LIST centroids coordinates
            self.old_centroids = np.empty_like(self.centroids) # LIST coordinates of centroids from previous iteration
            self.clusters = np.zeros(len(self.X))              # LIST list that assignes each element of X into a cluster
            self._cluster_points()                             # sets the first cluster assignation
        self.num_iter = 0                                      # INT current iteration
            
#############################################################
##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
#############################################################


    """ def _init_centroids(self):
        @brief Initialization of centroids
        depends on self.options['km_init']
        
#######################################################
##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
##  AND CHANGE FOR YOUR OWN CODE
#######################################################
        self.centroids= np.zeros((self.K,np.shape(self.X)[-1]))
        if self.options['km_init'].lower() == 'first':
	        self.centroids = np.sort(np.unique(self.X, axis=1).astype("float64")[:self.K]) #Coge los k primeros elementos
        else: #random
	        self.centroids = (np.random.rand(self.K, self.X.shape[1]).astype("float64")) * 255 """

    def _init_centroids(self):
        """@brief Initialization of centroids
        depends on self.options['km_init']
        """
        self.centroids = []
        
        if self.options['km_init'].lower() == 'first':
            # Estem agafant els indexs dels primers K punts de diferent color i 
            # utilitzant-los per a agafar aquests mateixos punts de la matriu X
            # Ho fem aixi perque np.unique retorna els valors ordenats i no ho volem
            # Utilitzem el np.sort perque np.unique no sempre retorna els index en ordre
            centroidIndexes = np.sort(np.unique(np.around(self.X), axis = 0, return_index = True)[1])[:self.K]
            if len(centroidIndexes) < self.K:
                nonUniqueIndex = 0
                while len(centroidIndexes) < self.K:
                    if nonUniqueIndex not in centroidIndexes:
                        centroidIndexes = np.append(centroidIndexes, nonUniqueIndex)
                    nonUniqueIndex += 1
            self.centroids = self.X[centroidIndexes]
        
        if self.options['km_init'].lower() == 'random':
            self.centroids = self.X[np.random.random_integers(0, self.X.shape[0]-1,self.K)]
        
        
    def _cluster_points(self):
        """@brief   Calculates the closest centroid of all points in X
        """
#######################################################
##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
##  AND CHANGE FOR YOUR OWN CODE
#######################################################
        distArray = distance(self.X, self.centroids)

        self.clusters = np.argmin(distArray, axis=1) #Retorna el indice del menor elemento de cada fila de distArray

        #self.clusters[i] -> centroide asignado al punto i
        
    def _get_centroids(self):
        """@brief   Calculates coordinates of centroids based on the coordinates 
                    of all the points assigned to the centroid
        """
#######################################################
##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
##  AND CHANGE FOR YOUR OWN CODE
#######################################################
	
        self.old_centroids = self.centroids.copy()


        clusterList = {}
        
        for i, point in enumerate(self.X):
            clusterList.setdefault(self.clusters[i], []).append(point)
        
        for i, cluster in clusterList.items():
            clusterArray = np.array(cluster, dtype="float64")
            self.centroids[i] = clusterArray.mean(axis=0)

    def _converges(self):
        """@brief   Checks if there is a difference between current and old centroids
        """
#######################################################
##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
##  AND CHANGE FOR YOUR OWN CODE
#######################################################
        return np.allclose(self.centroids, self.old_centroids, atol=self.options["tolerance"])
            #True si centroid[i] = old_centroid[i] dentro de la tolerancia, para todo i
        
    def _iterate(self, show_first_time=True):
        """@brief   One iteration of K-Means algorithm. This method should 
                    reassigne all the points from X to their closest centroids
                    and based on that, calculate the new position of centroids.
        """
        self.num_iter += 1
        self._cluster_points()
        self._get_centroids()
        if self.options['verbose']:
            self.plot(show_first_time)


    def run(self):
        """@brief   Runs K-Means algorithm until it converges or until the number
                    of iterations is smaller than the maximum number of iterations.=
        """
        if self.K==0:
            self.bestK()
            return        
        
        self._iterate(True)
        if self.options['max_iter'] > self.num_iter:
            while not self._converges() :
                self._iterate(False)
      
      
    def bestK(self):
        """@brief   Runs K-Means multiple times to find the best K for the current 
                    data given the 'fitting' method. In cas of Fisher elbow method 
                    is recommended.
                    
                    at the end, self.centroids and self.clusters contains the 
                    information for the best K. NO need to rerun KMeans.
           @return B is the best K found.
        """
#######################################################
##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
##  AND CHANGE FOR YOUR OWN CODE
#######################################################
        if self.options['fitting'].lower() == 'fisher':
            fit = [] #"Recta Fisher"
            for k in range(2,15):
                self._init_rest(k)
                self.run()        
                fit.append(self.fitting())
            
            fit2 = np.gradient(np.gradient(fit)) #Segona derivada "Recta Fisher"
            return np.argmax(abs(fit2)) + 2 #Maxim 2a deriv. => Colze. Minima K = 2
        
    def fitting(self):
        """@brief  return a value describing how well the current kmeans fits the data
        """
#######################################################
##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
##  AND CHANGE FOR YOUR OWN CODE
#######################################################
        if self.options['fitting'].lower() == 'fisher':
            intra=np.zeros(self.K)
            for k in range(self.K):
                a = distance(self.X[self.clusters==k,:],self.centroids)
                b = a[:,k]
                if b != []:
                    intra[k] = np.sum(b) / b.shape[0]
                
            a = self.centroids - np.mean(self.X, axis=0)
            a = np.sqrt(np.sum(a ** 2, axis=1))
            intraclass = np.sum(intra) / self.K
            interclass = np.sum(a) / (self.K)
            return intraclass/interclass
        else:
            return np.random.rand(1)


    def plot(self, first_time=True):
        """@brief   Plots the results
        """

        #markersshape = 'ov^<>1234sp*hH+xDd'	
        markerscolor = 'bgrcmybgrcmybgrcmyk'
        if first_time:
            plt.gcf().add_subplot(111, projection='3d')
            plt.ion()
            plt.show()

        if self.X.shape[1]>3:
            if not hasattr(self, 'pca'):
                self.pca = PCA(n_components=3)
                self.pca.fit(self.X)
            Xt = self.pca.transform(self.X)
            Ct = self.pca.transform(self.centroids)
        else:
            Xt=self.X
            Ct=self.centroids

        for k in range(self.K):
            plt.gca().plot(Xt[self.clusters==k,0], Xt[self.clusters==k,1], Xt[self.clusters==k,2], '.'+markerscolor[k])
            plt.gca().plot(Ct[k,0:1], Ct[k,1:2], Ct[k,2:3], 'o'+'k',markersize=12)

        if first_time:
            plt.xlabel('dim 1')
            plt.ylabel('dim 2')
            plt.gca().set_zlabel('dim 3')
        plt.draw()
        plt.pause(0.01)
