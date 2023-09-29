import requests
from pyquery import PyQuery
import matplotlib.pyplot as plt
import numpy as np
import random


# Takes in format { name: [x1, x2, x3...]  }
# Interprets number of dimensions from first coord
# Modifies original array and adds clusterID key

def kMeans(points=[], numDims=2, k=5):
    numDataPoints = len(points)
    clusterIDX = np.zeros(len(points))
    means = np.zeros((k, 2))
    prevMeans = None

    # Normalize data
    xyValues = np.array(list(points.values())).astype(int)
    xyValues = xyValues / np.max(xyValues.T, axis=1) 

    # Initialize means
    means = xyValues[random.sample(range(numDataPoints), k)]
    
    # Kmeans 
    while np.any(prevMeans == None) or np.all(np.linalg.norm(means - prevMeans, ord=2) < 0.01):
        # Get distance from each mean
        distances = np.linalg.norm(xyValues[:, np.newaxis] - means, ord=2, axis=2)
        # Get smallest distance and reassign each point
        clusterIDX = np.argmin(distances, axis=1)
        # Get new means
        prevMeans = np.copy(means)

        # Calculate new means
        for cluster in range(k):
            means[cluster] = np.mean(xyValues[clusterIDX == cluster], axis=0)

    return clusterIDX
            

def scrapeEveryNoise() -> dict:
    everyNoiseHtml = requests.get("http://everynoise.com")
    print(everyNoiseHtml)
    query = PyQuery(everyNoiseHtml.text)
    query = query('div.genre')
    genres = {}
    for div in query.items():
        for attr in div.attr('style').split('; '):
            key, val = attr.split(": ")
            if not genres.get(div.text()[:-1]):
                genres[div.text()[:-1]] = {}
            if key == 'top' or key == 'left':
                genres[div.text()[:-1]][key] = val[:-2]
            elif key == 'color':
                genres[div.text()[:-1]][key] = val
    for key, value in genres.items():
        coords = [value['left'], value['top']]
        genres[key] = coords
    return genres


if __name__ == "__main__":
    genres = scrapeEveryNoise()
    # Normalize data
    xyValues = np.array(list(genres.values())).astype(int)
    xyValues = xyValues / np.max(xyValues.T, axis=1) 

    
    plt.scatter(xyValues[:, 0], xyValues[:, 1], s=1, c=kMeans(genres))
    plt.savefig('./data.png')
