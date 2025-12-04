import heapq
import numpy as np

class Classifier:
    def __init__(self, n_neighbors=3):
        self.n_neighbors = n_neighbors
    
    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
    
    def distance(self, x, y):
        return np.sum((x - y) ** 2)
    
    def predict_single(self, x):
        heap = []
        
        for i in range(len(self.x_train)):
            dist = self.distance(self.x_train[i], x)

            heapq.heappush(heap, (-dist, self.y_train[i]))

            if len(heap) > self.n_neighbors:
                heapq.heappop(heap) 
        
        labels = [int(label) for (_, label) in heap]
        
        count = np.bincount(labels)
        return np.argmax(count)
    
    def predict(self, x_test):
        return np.array([self.predict_single(x) for x in x_test]).reshape(-1, 1)
