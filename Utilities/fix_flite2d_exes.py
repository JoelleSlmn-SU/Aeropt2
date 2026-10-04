"""
fix_flite2d_exes.py

Repair the 2013 builds of the FLITE 2D executables (Solver/PrePro.exe and
Solver/Solver.exe). Three defects, all found by running the programs under a
debugger; each fix is a few bytes in the executable file and the original is
kept next to it as <name>.exe.bak the first time.

Usage (from the repository root):
    python Utilities/fix_flite2d_exes.py Solver

The script only touches the exact builds it was written for: it checks the
file size and the bytes around each change first and refuses anything else.
Running it again is harmless ("already applied").

1. Stack limit (both programs): 1 MB -> 1024 MB.
   They keep work arrays sized by the number of mesh nodes on the stack. With
   the default 1 MB a large mesh stops at the first iteration with
       forrtl: severe (170): Program Exception - stack overflow

2. Solver.exe, one routine of the time-stepping loop (1 byte).
   It copies an array with 4 columns into one with 5 columns and takes the
   number of columns to copy from the DESTINATION, so it reads one whole
   column (4 bytes x number of nodes) past the end of the source. On a small
   mesh the stray read lands in neighbouring heap memory and goes unnoticed.
   Above roughly 65,000 nodes Windows gives the source array its own memory
   block with nothing readable behind it, and the solver stops with
       forrtl: severe (157): Program Exception - access violation
   Fix: take the column count from the SOURCE array
   (mov rcx,[rsi+0x6A8] -> mov rcx,[rsi+0x3A8]).
   Checked on a NACA 0012 mesh with 137,882 nodes (Mach 0.8, alpha 2 deg, 150
   multigrid cycles), original against fixed: the result file is identical
   byte for byte; residual, lift, drag and moment are identical. Only column
   6 of the .rsd ("pressure lift") differs: garbage in the original (it is
   fed by the stray read), equal to the lift in the fixed one, as it must be
   for inviscid flow.

3. PrePro.exe, the routine that builds the internal outflow register.
   When a mesh has no engine-inlet sides and no inflow/outflow chains - every
   plain external-flow case - the routine returns early and never sets the
   register's entry count. The count is then whatever was left in that piece
   of memory. If it happens to be zero or negative PrePro works; if it is
   positive PrePro tries to write that many entries of an array that was
   never allocated and stops, after "Writing grid data # 1 to file...", with
       forrtl: severe (157): Program Exception - access violation
   leaving a partial .sol file behind. Whether it happens depends on things
   as unrelated as the environment variables of the process, so the same
   mesh can work in one session and fail in another.
   Fix: set the count to zero on that early return (a jump to 18 bytes of
   new code placed in the unused tail of the code section).
   Checked: the fixed PrePro writes a .sol identical byte for byte to the one
   the original writes when it happens to work (137,882-node mesh), and it
   completes on three meshes on which the original crashed.
"""
import os
import shutil
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from set_exe_stack import set_stack_reserve, stack_reserve      # noqa: E402

STACK_MB = 1024

# name -> (file size, [(description, file offset, original bytes, new bytes), ...])
_BUILDS = {
    "Solver.exe": (1309696, [
        ("array-copy fix",
         0x634C,
         bytes.fromhex("488b9690060000" "488b8ea8060000" "440f29bd70020000"),
         bytes.fromhex("488b9690060000" "488b8ea8030000" "440f29bd70020000")),
    ]),
    "PrePro.exe": (818688, [
        ("outflow-register count fix: jump",          # mov [rbx+0x170],rax -> jmp to the new code
         0x8B68,
         bytes.fromhex("33c0" "48898358010000" "48898370010000" "4881c4f8010000"),
         bytes.fromhex("33c0" "48898358010000" "e92a640900" "9090" "4881c4f8010000")),
        ("outflow-register count fix: new code",      # mov [rbx+0x170],rax; mov [rbx+0x1EC],eax; jmp back
         0x9EFA0,
         bytes(18),
         bytes.fromhex("48898370010000" "8983ec010000" "e9c69bf6ff")),
        ("outflow-register count fix: section size",  # .text virtual size covers the new code
         0x1F8,
         struct.pack("<I", 0x9EB91),
         struct.pack("<I", 0x9EBB2)),
    ]),
}


def fix_exe(path: str) -> list:
    """Apply the fixes for this executable; returns lines saying what was done."""
    name = os.path.basename(path)
    key = next((k for k in _BUILDS if k.lower() == name.lower()), None)
    if key is None:
        raise ValueError(f"{name}: no fixes known for a file of this name.")
    size, patches = _BUILDS[key]
    with open(path, "rb") as fh:
        data = bytearray(fh.read())
    if len(data) != size:
        raise ValueError(f"{name}: not the build the fixes were written for "
                         f"(size {len(data)}, expected {size}); nothing was changed.")
    state = []
    for what, off, old, new in patches:
        cur = bytes(data[off:off + len(old)])
        if cur not in (old, new):
            raise ValueError(f"{name}: unexpected bytes at offset {off:#x} ({what}); nothing was changed.")
        state.append(cur == new)
    done = []
    if all(state):
        done.append(f"{name}: code fix already applied")
    else:
        bak = path + ".bak"
        if not os.path.exists(bak):
            shutil.copy2(path, bak)
        for what, off, old, new in patches:
            data[off:off + len(old)] = new
        tmp = path + ".tmp"
        with open(tmp, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
        done.append(f"{name}: code fix applied ({patches[0][0].split(':')[0]})")
    before = stack_reserve(path)
    if before < STACK_MB * 2 ** 20:
        set_stack_reserve(path, STACK_MB)
        done.append(f"{name}: stack limit {before / 2 ** 20:g} MB -> {STACK_MB} MB")
    else:
        done.append(f"{name}: stack limit already {before / 2 ** 20:g} MB")
    return done


def is_fixed(path: str) -> bool:
    """True if every code fix for this executable is in place (False if the file is unknown)."""
    name = os.path.basename(path)
    key = next((k for k in _BUILDS if k.lower() == name.lower()), None)
    if key is None:
        return False
    size, patches = _BUILDS[key]
    try:
        with open(path, "rb") as fh:
            data = fh.read()
    except OSError:
        return False
    return len(data) == size and all(data[off:off + len(new)] == new for _w, off, _old, new in patches) \
        and stack_reserve(path) >= STACK_MB * 2 ** 20


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    target = sys.argv[1]
    files = ([os.path.join(target, n) for n in ("PrePro.exe", "Solver.exe")] if os.path.isdir(target)
             else [target])
    failed = False
    for f in files:
        try:
            for line in fix_exe(f):
                print(line)
        except (OSError, ValueError) as e:
            failed = True
            print(f"ERROR: {e}")
    print("Originals are kept as <name>.exe.bak")
    sys.exit(1 if failed else 0)
