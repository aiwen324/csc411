from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
import os
from scipy.ndimage import filters
import urllib

# Split by tab so it actually parse the whole name, not only first name
# act = list(set([a.split("\t")[0] for a in open("facescrub_actresses.txt").readlines()]))

# actors for the first several parts
male_actors = ['baldwin', 'hader', 'radcliffe', 'butler', 'vartan', 'carell']
actresses = ['bracco', 'chenoweth', 'drescher', 'ferrera', 'gilpin', 'harmon']
act1 = ['Peri Gilpin', 'Angie Harmon']
act2 = ['Alec Baldwin', 'Bill Hader', 'Steve Carell']
test_actress = ['Lorraine Bracco']
act3_male = ['Daniel Radcliffe', 'Gerard Butler', 'Michael Vartan']
act4_female = ['Kristin Chenoweth', 'Fran Drescher', 'America Ferrera']
act5 = ['Daniel Radcliffe', 'Gerard Butler', 'Michael Vartan', 'Kristin Chenoweth', 'Fran Drescher', 'America Ferrera']

def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    '''From:
    http://code.activestate.com/recipes/473878-timeout-function-using-threading/'''
    import threading
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        return False
    else:
        return it.result

testfile = urllib.URLopener()            


# Note: you need to create the uncropped folder first in order 
# for this to work


"""
# test parse
for a in test_actress:
    # Parsing Lorraine Bracco
    name = a.split()[1].lower()
    fname = name + ".txt"
    f = open(fname, "w")
    i = 0
    for line in open("facescrub_actresses.txt"):
        if a in line:
            filename = name+str(i)+'.'+line.split()[4].split('.')[-1]
            #A version without timeout (uncomment in case you need to 
            #unsupress exceptions, which timeout() does)
            #testfile.retrieve(line.split()[4], "uncropped/"+filename)
            #timeout is used to stop downloading images which take too long to download
            timeout(testfile.retrieve, (line.split()[4], "uncropped/"+filename), {}, 30)
            if not os.path.isfile("uncropped/"+filename):
                continue
            print filename
            line_to_wrt = filename + " " + line.split()[1] + "\t" + line.split()[4] + "\t" + line.split()[5] + "\n"
            f.write(line_to_wrt)
            i += 1
    f.close()


"""
# Parse actress
for a in act4_female:
    # Parsing Lorraine Bracco
    name = a.split()[1].lower()
    if not os.path.exists("uncropped/"+name) or not os.path.isdir("uncropped/"+name):
        os.mkdir("uncropped/"+name)
    fname = str(name) + ".txt"
    f = open(fname, "w")
    i = 0
    for line in open("facescrub_actresses.txt"):
        if a in line:
            filename = name+str(i)+'.'+line.split()[4].split('.')[-1]
            #A version without timeout (uncomment in case you need to 
            #unsupress exceptions, which timeout() does)
            #testfile.retrieve(line.split()[4], "uncropped/"+filename)
            #timeout is used to stop downloading images which take too long to download
            timeout(testfile.retrieve, (line.split()[4], "uncropped/"+name+"/"+filename), {}, 30)
            if not os.path.isfile("uncropped/"+name+"/"+filename):
                continue
            print filename
            line_to_wrt = line.split()[1] + " " + filename + "\t" + line.split()[4] + "\t" + line.split()[5] + '\n'
            f.write(line_to_wrt)
            i += 1
    f.close()

# Parse actors
for a in act3_male:
    # Parsing Lorraine Bracco
    name = a.split()[1].lower()
    if not os.path.exists("uncropped/"+name) or not os.path.isdir("uncropped/"+name):
        os.mkdir("uncropped/"+name)
    fname = str(name) + ".txt"
    f = open(fname, "w")
    i = 0
    for line in open("facescrub_actors.txt"):
        if a in line:
            filename = name+str(i)+'.'+line.split()[4].split('.')[-1]
            #A version without timeout (uncomment in case you need to 
            #unsupress exceptions, which timeout() does)
            #testfile.retrieve(line.split()[4], "uncropped/"+filename)
            #timeout is used to stop downloading images which take too long to download
            timeout(testfile.retrieve, (line.split()[4], "uncropped/"+name+"/"+filename), {}, 30)
            if not os.path.isfile("uncropped/"+name+"/"+filename):
                continue
            print filename
            line_to_wrt = line.split()[1] + " " + filename + "\t" + line.split()[4] + "\t" + line.split()[5] + '\n'
            f.write(line_to_wrt)
            i += 1
    f.close()