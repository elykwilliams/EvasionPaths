#!/usr/bin/env python3

import sys
import pickle
import pprint

def load_dict(fn):

    with open(fn, 'rb') as f:
        data = pickle.load(f)

    print(data)
if __name__ == "__main__":
    fn = "./output/Tmax/tmax.pkl"
    load_dict(fn)
