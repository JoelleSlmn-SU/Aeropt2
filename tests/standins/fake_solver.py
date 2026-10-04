"""Stand-in for Solver.exe in the tests. The Mach number selects the behaviour:
   9.99 -> NaN from the first iteration, 8.88 -> never finishes, 7.77 -> stops without output,
   6.66 -> one iteration, then a Fortran run-time error.
   Like the real solver it rewrites the result file every ivd%writeToFileInterval iterations
   and once more at the end, one row per mesh node, the node number in a 5-character field.
   Started with the extra argument --geometry, lift and drag also depend on the wall shape the
   stand-in preprocessor recorded: drag has its minimum at a mean |y| of 0.03."""
import os
import re
import sys
import time

ans = []
for p in ["Enter control filename: ", "Enter computation filename: ", "Enter startup filename: ",
          "Enter result filename: ", "Enter residual filename: "]:
    sys.stdout.write(p)
    ans.append(sys.stdin.readline().rstrip("\n"))
control, comp, startup, res, rsd = ans
print("\n WELCOME TO MGNS2D (stand-in)")
for f in (control, comp):
    if not os.path.isfile(f):
        print(f"File Open Error: {f}")
        sys.exit(2)
assert startup == "", repr(startup)
text = open(control).read()
val = lambda name: float(re.search(r"ivd%" + name + r"\s*=\s*([^,\n]+)", text).group(1))   # noqa: E731
mach, alpha = val("MachNumber"), val("alpha")
interval = int(val("writeToFileInterval"))
with open(comp) as fh:                       # the stand-in .sol: "grids n", "ne np nb", "shape t c"
    sol = fh.read().splitlines()
n_nodes = int(sol[1].split()[1])
lift, drag = 0.11 * alpha, 0.001 * mach
if "--geometry" in sys.argv:
    thick, camber = (float(v) for v in sol[2].split()[1:3])
    lift += 5.0 * camber
    drag += 10.0 * (thick - 0.03) ** 2


def write_res(it, nan=False):
    """A smooth made-up field around the free stream (rho, u, v, E, p), different at every node."""
    p_inf = 1.0 / (1.4 * mach * mach)
    with open(res, "w") as fh:
        for k in range(1, n_nodes + 1):
            w = 0.1 * (k % 17) / 17.0 + 0.001 * (it % 7)
            row = [1.0 + 0.2 * w, 1.0 - w, 0.3 * w, p_inf / 0.4 + 0.5, p_inf * (1.0 + 0.5 * w)]
            num = f"{k:5d}" if k < 100000 else "*****"
            fh.write(num + "".join("           NaN" if nan else f"{v:14.4E}" for v in row) + "\n")


if mach == 7.77:
    print("ERROR: Wrong boundary term")
    sys.exit(0)
with open(rsd, "w") as fh:
    if mach == 9.99:
        fh.write("      1           NaN           NaN           NaN           NaN           NaN           NaN   0.00000E+00\n")
    else:
        n = 10 ** 9 if mach == 8.88 else 5
        if mach == 6.66:                 # a few good iterations, then a run-time error
            fh.write("      1  -0.10000E+00   0.10000E+00   0.10000E-01   0.00000E+00\n")
            fh.flush()
            print("forrtl: severe (157): Program Exception - access violation")
            sys.stdout.flush()
            os._exit(157)
        i = 0
        while i < n:
            i += 1
            fh.write(f"{i:7d} {-0.5 * i:13.5E} {lift:13.5E} {drag:13.5E} {0.0:13.5E} "
                     f"{0.0:13.5E} {0.0:13.5E} {0.0:13.5E}\n")
            fh.flush()
            if interval > 0 and i % interval == 0:
                write_res(i)
            if mach == 8.88:
                time.sleep(0.05)
write_res(0, nan=(mach == 9.99))
print("Final residual")
