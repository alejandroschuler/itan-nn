#!/usr/bin/env python

import sys

last_enc = None
encounters = {}

for line in sys.stdin:
    details = line.split("\t")
    enc = details[2]

    if enc not in encounters.keys():
        encounters[enc] = [details]
    else:
        encounters[enc].append(details)

for enc, rows in encounters.items():
    for row in rows:
        print('\t'.join(row))
