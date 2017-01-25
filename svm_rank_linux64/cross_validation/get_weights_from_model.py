# Compute the weight vector of linear SVM based on the model file
# Original Perl Author: Thorsten Joachims (thorsten@joachims.org)
# Python Version: Ori Cohen (orioric@gmail.com)
# Call: python svm2weights.py svm_model

import sys
from operator import itemgetter

try:
    import psyco
    psyco.full()
except ImportError:
    print 'Psyco not installed, the program will just run slower'

def sortbyvalue(d,reverse=True):
    ''' proposed in PEP 265, using  the itemgetter this function sorts a dictionary'''
    return sorted(d.iteritems(), key=itemgetter(1), reverse=True)

def sortbykey(d,reverse=True):
    ''' proposed in PEP 265, using  the itemgetter this function sorts a dictionary'''
    return sorted(d.iteritems(), key=itemgetter(0), reverse=False)

def get_file():
    """
    Tries to extract a filename from the command line.  If none is present, it
    assumes file to be svm_model (default svmLight output).  If the file 
    exists, it returns it, otherwise it prints an error message and ends
    execution. 
    """
    # Get the name of the data file and load it into 
    if len(sys.argv) < 2:
        # assume file to be svm_model (default svmLight output)
        print "Usage: python get_weights_from_model/model_name"
        sys.exit()
    else:
        filename = sys.argv[1]

    
    try:
        f = open(filename, "r")
    except IOError:
        print "Error: The file '%s' was not found on this system." % filename
        sys.exit(0)

    return f




if __name__ == "__main__":
    f = get_file()
    i=0
    lines = f.readlines()
    printOutput = True
    w = {}
    for line in lines:
        if i>10:
            features = line[:line.find('#')-1]
            comments = line[line.find('#'):]
            alpha = features[:features.find(' ')]
            feat = features[features.find(' ')+1:]
            for p in feat.split(' '): # Changed the code here. 
                a,v = p.split(':')
                if not (int(a) in w):
                    w[int(a)] = 0
            for p in feat.split(' '): 
                a,v = p.split(':')
                w[int(a)] +=float(alpha)*float(v)
        elif i==1:
            if line.find('0')==-1:
                print 'Not linear Kernel!\n'
                printOutput = False
                break
        elif i==10:
            if line.find('threshold b')==-1:
                print "Parsing error!\n"
                printOutput = False
                break
        
        i+=1    
    f.close()

    #if you need to sort the features by value and not by feature ID then use this line intead:
    #ws = sortbyvalue(w) 
    
    ws = sortbykey(w)
    if printOutput == True:
        for (i,j) in ws:
            print i,':',j
            i+=1
