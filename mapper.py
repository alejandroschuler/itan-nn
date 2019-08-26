
#!/usr/bin/env python

import sys

next(sys.stdin) # skip the header row
for line in sys.stdin:
    line = line.strip()
    keys = line.split("\t");
    keys = [ keys[2] ] + keys
    print("\t".join(str(v) for v in keys))