import numpy as np


class KMeans():

    #NOTE: CHANGE max_iter to 300
    def __init__(self, n_clusters: int, init: str='random', max_iter = 300):
        """

        :param n_clusters: number of clusters
        :param init: centroid initialization method. Should be either 'random' or 'kmeans++'
        :param max_iter: maximum number of iterations
        """
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.centroids = None # Initialized in initialize_centroids()

    def fit(self, X: np.ndarray):
        self.centroids = self.initialize_centroids(X)
        # print(self.centroids.shape)
        iteration = 0
        clustering = []
        
        
        while iteration < self.max_iter:
            # your code
            clustering = []
            for i in range(0,self.n_clusters):
                clustering.append([])
            for cell in X:
                #Determine nearest centroid
                dist_matrices = []
                for i in range(0,self.n_clusters):
                    dist_matrices.append(self.euclidean_distance(cell,self.centroids[i]))
                
                dist_index = 0
                min_index = -1
                while dist_index < self.n_clusters:
                    if min_index == -1:
                        min_index = dist_index
                    elif dist_matrices[dist_index][0] < dist_matrices[min_index][0]:
                        min_index = dist_index
                    dist_index = dist_index + 1
            
                #Debugging purposes
                if min_index < 0:
                    raise ValueError("Error in fit: min_index value is invalid")
                
                #Create clustering accordingly
                clustering[min_index].append(cell)
            
            #Update centroids
            old_centroids = self.centroids
            self.centroids = self.update_centroids(clustering, X)
            # print(self.centroids.shape)
            # print(self.centroids)
            
            #Termination condition
            isComplete = True
            # print("----")
            # print("old: ", old_centroids)
            # print("new: ", self.centroids)
            for i in range(0,self.centroids.shape[0]):
                if not(self.euclidean_distance(self.centroids[i], old_centroids[i]) == 0):
                    isComplete = False
            if isComplete:
                break              
                
            iteration = iteration + 1
            
        print("iterations: ", iteration)
        return clustering

    def update_centroids(self, clustering: np.ndarray, X: np.ndarray):
        #your code
        num_clusters = self.n_clusters
        
        #Check if clustering is of right size (Assuming n_clusters > 1)
        if not(len(clustering) == num_clusters):
            raise ValueError("Error in update_centroids: clustering array is not of correct shape")
        
        updated_centroids = np.array([])
        
        for cluster in clustering:
            if (not(str(type(cluster[0])) == "<class 'numpy.ndarray'>")):
                updated_centroids = np.append(updated_centroids, np.mean(cluster))
            else:
                if updated_centroids.shape[0] == 0:
                    updated_centroids = np.append(updated_centroids, np.mean(cluster, axis=0))
                else:
                    updated_centroids = np.vstack([updated_centroids, np.mean(cluster, axis=0)])
                    
        return updated_centroids

    def initialize_centroids(self, X: np.ndarray):
        """
        Initialize centroids either randomly or using kmeans++ method of initialization.
        :param X:
        :return:
        """
        X_copy = X[:]
        num_clusters = self.n_clusters
        centroids = np.array([]) #Initialize centroids array
        
        if self.init == 'random':
            # your code
            indexes = np.array([])
            for i in range(0,X.shape[0]):
                indexes = np.append(indexes, int(i))
            
            centroids_indexes = np.random.choice(indexes, size=num_clusters, replace=False)
            for i in centroids_indexes:
                if centroids.shape[0] == 0:
                    centroids = np.append(centroids, X[int(i)])
                else:
                    centroids = np.vstack([centroids, X[int(i)]])
            
        elif self.init == 'kmeans++':
            #NOTE: COMPLETE THIS
            # randomly selecting centroids
            indexes = np.array([])
            for i in range(0,X.shape[0]):
                indexes = np.append(indexes, int(i))
            
            centroids_indexes = np.random.choice(indexes, size=num_clusters, replace=False)
            for i in centroids_indexes:
                if centroids.shape[0] == 0:
                    centroids = np.append(centroids, X[int(i)])
                else:
                    centroids = np.vstack([centroids, X[int(i)]])
                    
            new_centroids = np.array([])
            
            while new_centroids.size < centroids.size:
                #Compute distances from nearest centroids
                min_distances = []
                for cell in X_copy:
                    distances = []
                    for c in centroids:
                        distances.append(int(self.euclidean_distance(cell, c)))
                    
                    min_dist = min(distances)
                    min_distances.append(min_dist)
                    
                total = sum(min_distances)
                p = min_distances[:]
                for i in range(0,len(p)):
                    p[i] = p[i]/total
                    
                # print("-----")
                # print(min_distances)
                # print(p)
                
                indexes = []
                for i in range(0,X_copy.shape[0]):
                    indexes.append(i)
                
                new_index = int(np.random.choice(indexes, size=1,p=p ,replace=False))
                # if new_centroids.shape[0] == 0:
                #     new_centroids = np.append([])
                # new_centroids = np.append(new_centroids, X_copy[new_index])
                
                if new_centroids.shape[0] == 0:
                    new_centroids = np.append(new_centroids, X_copy[new_index])
                else:
                    new_centroids = np.vstack([new_centroids, X_copy[new_index]])
                
                # print(new_index)
                # print("-----")
                if new_index == indexes[-1]:
                    X_copy = X_copy[:-1]
                else:
                    X_copy = np.vstack([X_copy[:new_index], X_copy[new_index+1:]])
                    
            centroids = new_centroids[:]
                    
            
        
        else:
            raise ValueError('Centroid initialization method should either be "random" or "k-means++"')
            
        return centroids

    def euclidean_distance(self, X1:np.ndarray, X2:np.ndarray):
        """
        Computes the euclidean distance between all pairs (x,y) where x is a row in X1 and y is a row in X2.
        Tip: Using vectorized operations can hugely improve the efficiency here.
        :param X1:
        :param X2:
        :return: Returns a matrix `dist` where `dist_ij` is the distance between row i in X1 and row j in X2.
        """
        #Check whether matrices have same shape
        if not(X1.shape == X2.shape):
            print("Error computing euclidean_distance: The arrays are of unequal shapes...")
            return np.array([])
        
        #Split matrices into vectors
        num_rows = X1.shape[0]
        
        if X1.shape[0] == X1.size:
            x1_vectors = X1
            x2_vectors = X2
        else:
            x1_vectors = np.vsplit(X1, num_rows)
            x2_vectors = np.vsplit(X2, num_rows)
        
        #For debugging purposes
        
        #Initialize distance matrix
        dist_matrix = np.array([])
        
        #Populate dist_matrix
        if X1.shape[0] == X1.size:
            dist = np.linalg.norm(x1_vectors - x2_vectors)
            return np.array([dist])
        
        for i in x1_vectors:    
            dist_vector = np.array([])
            for j in x2_vectors:
                dist = np.linalg.norm(i - j)
                dist_vector = np.append(dist_vector, dist)
            if dist_matrix.shape[0] == 0:
                dist_matrix = np.append(dist_matrix, dist_vector)
            else:
                dist_matrix = np.vstack([dist_matrix, dist_vector])
        
        return dist_matrix
        
        
    def silhouette(self, clustering: np.ndarray, X: np.ndarray):
        centroids = self.centroids
        
        silhouette_list = []
        
        for cluster in clustering:
            for cell in cluster:
                distances = []
                for c in centroids:
                    distances.append(int(self.euclidean_distance(cell,c)))
                    
                a = min(distances)
                distances.remove(a)
                b = min(distances)
                
                s = (b-a)/max(a,b)
                silhouette_list.append(s)
                
        silhouette_coeff = np.mean(silhouette_list)
        return silhouette_coeff