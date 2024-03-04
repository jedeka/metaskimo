import numpy as np
import os, json, glob 

import matplotlib.pyplot as plt 

jsons_ms = sorted(glob.glob('json_logs/metaskimo/seed*.json'))  # change this dir

# metaskimo

files = []

for jsonfile in jsons_ms:
    with open(jsonfile, 'r+') as f:
        file = json.load(f)
        files.append(file)

smooths = []
for i, file in enumerate(files):
    # task = i % 10
    task = jsons_ms[i].split('.')[0][-1]
    smoothing_weight = 0.99
    
    print(task)
    value = np.array(file[str(task)])[:500]
    
    last = value[:10].mean()  # First value in the plot (first timestep)
    
    print(last, len(file[task]))
    
    # print(last, values[seed][:10])
    smoothed = list()
    for point in value:
        smoothed_val = (
            last * smoothing_weight + (1 - smoothing_weight) * point
        )  # Calculate smoothed value
        smoothed.append(smoothed_val)  # Save it
        last = smoothed_val  # Anchor the last smoothed value

    smooths.append(smoothed)
    
x_ms = np.mean(smooths, axis=0)



plt.plot(x_ms)
plt.savefig('plot.png')

