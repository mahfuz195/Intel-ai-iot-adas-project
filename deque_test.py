import collections
import numpy as np

right = collections.deque(maxlen=5)

for i in range(6):
    right.append([i,i+1, i+2])

print (right)


print (np.mean(right,axis=0).astype(int))
right.append([11,12,14])
print (right)
