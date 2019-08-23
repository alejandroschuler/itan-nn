#!/usr/bin/env python

import sys, csv

for line in sys.stdin:
    line = line.strip()
    keys = line.split("\t");
    keys = [ keys[2] ] + keys  # Put the key for the reducer at the front of the line--the third element is the encounter Id
    print("\t".join(str(v) for v in keys ) )  # Output in tab-delimited with encounter Id at the beginning
