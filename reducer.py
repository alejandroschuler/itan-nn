#!/usr/bin/env python

import sys

# Example input (ordered by key# keys come grouped together
# so we need to keep track of state a little bit
# thus when the key changes (enc), we need to reset
# our counter, and write out the count we've accumulated

last_enc = None
encounters = {}
datePosition = 6

for line in sys.stdin:
    line = line.strip()
    details = line.split("\t")
    enc = details[0]

    # if this is the first iteration
    if not last_enc:
        last_enc = enc

    if enc != last_enc:
        # state change (previous line was k=x, this line is k=y)
        for key in sorted(encounters.keys()):
            print("\t".join(str(v) for v in encounters[key]))
        last_enc = enc
        encounters = {}

    encounters[details[datePosition]] = details

# this is to catch the final counts after all records have been received.
# print("\t".join(str(v) for v in details ) )
