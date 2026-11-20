import numpy as np

def vin_dropout(vin_x, p=0.5, training=True):

    vin_output = []

    if training:
        for i in range(len(vin_x)):
            if np.random.rand() > p:
                vin_output.append(vin_x[i] / (1-p))
            else:
                vin_output.append(0)
    else:
        vin_output = vin_x

    return vin_output

vin_x = [2, 4, 6, 8, 10]

vin_result = vin_dropout(vin_x, p=0.5)

print("Input:", vin_x)
print("Output:", vin_result)

#############################################

import numpy as np

def vin_dropout(vin_x,p = 0.5):
    vin_mask = np.random.randint(0,2,len(vin_x))
    vin_output = vin_x * vin_mask/ p
    return vin_output

vin_x = np.array([2,4,6,8,10])

print(vin_dropout(vin_x))
