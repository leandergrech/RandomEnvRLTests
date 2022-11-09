import numpy as np
from datetime import datetime as dt

start = dt.now()
for _ in range(1000):
    max(np.random.normal(0, 1, 3**7))
end = dt.now()
t1 = end - start

start  = dt.now()
for _ in range(1000):
    max(np.random.normal(0, 1, 15))
end = dt.now()
t2 = end - start

print(t1/t2)
print(3**7/15)
