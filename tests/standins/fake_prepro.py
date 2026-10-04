"""Stand-in for PrePro.exe in the tests: same prompts, same answers from stdin."""
import os
import sys

def ask(prompt):
    sys.stdout.write(prompt)
    return sys.stdin.readline().rstrip("\n")


# same questions, in the same order, as the real PrePro.exe
mesh = ask("Enter input filename: ")
hybrid = ask("Is grid hybrid (.true./.false.)/(T/F): ")
ngrid = ask("Number of grids to generate: ")
direc = ask("Directionality parameter: ")
ndir = ask("Number of directional agglomerations: ") if float(direc) != 0.0 else "0"
vis = ask("Visualization mode (0: none): ")
assert vis == "0", f"visualization mode {vis!r}: the real program would now ask for more files"
out = ask("Computation output file: ")
print("\n WELCOME TO AGGL2D (stand-in)")
if not os.path.isfile(mesh):
    print(f"File Open Error: {mesh}")
    sys.exit(2)
assert hybrid in (".true.", ".false."), hybrid
if int(ngrid) > 9:
    print("To many grids to be generated, coarse grid has only one node!")
    sys.exit(0)                      # like a Fortran STOP: exit code 0, no output file
float(direc), int(ndir)
if int(ngrid) == 7:                  # never finishes (to test stopping the preprocessor)
    sys.stdout.flush()
    import time
    while True:
        time.sleep(0.05)
with open(mesh) as fh:
    rows = fh.read().splitlines()
head = rows[:4]
# two numbers describing the wall shape, so the stand-in solver can return forces that
# depend on the design: mean |y| ("thickness") and mean y ("camber") of the wall nodes
ne, npn, nb = (int(v) for v in head[3].split()[:3])
xy = {int(r.split()[0]): (float(r.split()[1]), float(r.split()[2])) for r in rows[6 + ne:6 + ne + npn]}
wall = sorted({int(t) for r in rows[7 + ne + npn:7 + ne + npn + nb] if r.split()[2] in ("1", "5")
               for t in r.split()[:2]})
thick = sum(abs(xy[k][1]) for k in wall) / max(len(wall), 1)
camber = sum(xy[k][1] for k in wall) / max(len(wall), 1)
with open(out, "w") as fh:
    fh.write(f"grids {ngrid}\n" + head[3] + "\n" + f"shape {thick!r} {camber!r}\n")
    fh.flush()
    if int(ngrid) == 8:              # dies while writing, like a real access violation
        print(" Writing grid data #           1  to file...")
        print("forrtl: severe (157): Program Exception - access violation")
        sys.stdout.flush()
        os._exit(157)
print("Agglomerating grid # done")
