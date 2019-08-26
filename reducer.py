#!/usr/bin/env python

import sys

last_enc = None
encounters = {}

for line in sys.stdin:
    details = line.split("\t")
    enc = details[0]
    row = details[1:]

    if enc not in encounters.keys():
        encounters[enc] = [row]
    else:
        encounters[enc].append(row)

for enc, rows in encounters.items():
    for row in rows:
        print('\t'.join(row))
