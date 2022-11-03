from collections import OrderedDict
import pickle

def func(a):
    return a + 1

def test1():        
    od = OrderedDict()
    od['a'] = 3
    b = pickle.dumps(od)
    s = ''
    for c in b:
        s += str(c) + ','
    return s[:-1]

def test2(s):
    l = []
    for c in s.split(','):
        l.append(int(c))
    b = bytes(l)
    od = pickle.loads(b)
    with open('test.txt', 'w') as f:
        f.write(str(od))
