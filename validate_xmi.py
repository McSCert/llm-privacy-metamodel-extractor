import sys
from pathlib import Path
from pyecore.resources import ResourceSet, URI

rset = ResourceSet()
pkg = rset.get_resource(URI("metamodel/privacy_metamodel.ecore")).contents[0]
rset.metamodel_registry[pkg.nsURI] = pkg

bad = 0
for f in sys.argv[1:]:
    try:
        root = rset.get_resource(URI(f)).contents[0]
        n = sum(1 for _ in root.eAllContents())
        print(f"  ok    {Path(f).name[:52]:<52} {n} elements")
    except Exception as exc:
        bad += 1
        print(f"  FAIL  {Path(f).name[:52]:<52} {type(exc).__name__}: {exc}")
sys.exit(1 if bad else 0)