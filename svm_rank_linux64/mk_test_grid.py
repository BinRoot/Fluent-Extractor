import numpy as np

with open('my_grid_test.dat', 'w') as writer:
    for w in np.arange(0, 1, 0.02):
        for h in np.arange(0, 1, 0.02):
#            line = "1 qid:5 1:{} 2:{} 3:{} 4:{} 5:{}\n".format(w, h, w*w, h*h, w*h)
            line = "1 qid:5 1:{} 2:{}\n".format(w, h)
            writer.write(line)
    
