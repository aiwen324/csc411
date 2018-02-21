import os
import urllib
import hashlib
male_actors = ['baldwin', 'hader', 'radcliffe', 'butler', 'vartan', 'carell']
actresses = ['bracco', 'chenoweth', 'drescher', 'ferrera', 'gilpin', 'harmon']
male_actors_fullname = ['Alec Baldwin', 'Bill Hader', 'Daniel Radcliffe', 'Gerard Butler', 'Michael Vartan', 'Steve Carell']
actresses_fullname = ['Lorraine Bracco', 'Kristin Chenoweth', 'Fran Drescher', 'America Ferrera', 'Peri Gilpin', 'Angie Harmon']
test_actors = ['Alec Baldwin']


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



def sha256_checksum(filename, block_size=65536):
    sha256 = hashlib.sha256()
    with open(filename, 'rb') as f:
        for block in iter(lambda: f.read(block_size), b''):
            sha256.update(block)
    return sha256.hexdigest()


testfile = urllib.URLopener()

for a in male_actors_fullname:
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
            filepath = "uncropped/"+name+"/"+filename
            timeout(testfile.retrieve, (line.split()[4], filepath), {}, 30)
            if not os.path.isfile("uncropped/"+name+"/"+filename):
                continue
            print filename
            hashword = sha256_checksum(filepath)
            if hashword != line.split()[6]:
                os.remove(filepath)
                print "Detecting unmatched HASH!!!!!!"
                continue
            line_to_wrt = line.split()[1] + " " + filename + "\t" + line.split()[4] + "\t" + line.split()[5] + '\n'
            f.write(line_to_wrt)
            i += 1
    f.close()