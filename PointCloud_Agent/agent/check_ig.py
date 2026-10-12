# -*- coding: utf-8 -*-
"""
Check that ig files really contain event content.

Catches the failure mode where every event only holds Event_V3 + Errors_V1
("... are not found") because the input had none of the expected collections.
Exit code 1 if Tracks_V3 is empty in every event (or there are no events).

Works with python2 and python3, standard library only.

  python3 check_ig.py NAME_0.ig [NAME_1.ig ...]
"""
from __future__ import print_function

import json
import re
import sys
import zipfile
from collections import Counter

# collections the point cloud uses; tuples = any of these names
NEEDED = ["Tracks_V3", "Extras_V1", "GsfElectrons_V2", "TrackerMuons_V2",
          "GlobalMuons_V1", "StandaloneMuons_V2", ("PrimaryVertices_V1", "Vertices_V1"),
          "EBRecHits_V2", "EERecHits_V2", "HBRecHits_V2"]


def load_event(z, name):
    text = z.read(name).decode("utf-8", "replace")
    text = re.sub(r'-?\b(?:nan|NaN|Infinity|inf)\b', 'null', text)
    return json.loads(text)


def label(c):
    return c if isinstance(c, str) else "/".join(c)


def main(paths):
    bad = False
    for path in paths:
        z = zipfile.ZipFile(path)
        events = [n for n in z.namelist() if "Events/" in n and not n.endswith("/")]
        prefixed = [n for n in events if not n.startswith("Events/")]
        print("=== %s : %d events" % (path, len(events)))
        if prefixed:
            print("  [!] entry names carry a path prefix (outputFilePath was set): %s" % prefixed[0])

        counts = Counter()
        errors = Counter()
        for n in events:
            coll = load_event(z, n).get("Collections", {})
            for c in NEEDED:
                names = (c,) if isinstance(c, str) else c
                if any(len(coll.get(x) or []) > 0 for x in names):
                    counts[label(c)] += 1
            for e in coll.get("Errors_V1") or []:
                errors[str(e[0])[:120]] += 1

        for c in NEEDED:
            k = label(c)
            mark = "OK " if counts[k] > 0 else "-- "
            print("  [%s] %-34s non-empty in %d/%d events" % (mark, k, counts[k], len(events)))
        if errors:
            print("  ISpy error messages (top 10):")
            for msg, k in errors.most_common(10):
                print("    %3dx %s" % (k, msg))
        if not events or counts["Tracks_V3"] == 0:
            print("  [!] no tracks in any event -> conversion failed")
            bad = True
    return 1 if bad else 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    sys.exit(main(sys.argv[1:]))
