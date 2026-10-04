"""
set_exe_stack.py

Show or raise the stack size of a Windows executable (the "stack reserve" in
its PE header). Same effect as Microsoft's  editbin /STACK:<bytes> file.exe,
without needing Visual Studio.

Why: the FLITE 2D Solver.exe was built with the default 1 MB stack. It keeps
work arrays sized by the number of mesh nodes on the stack, so a large mesh
stops at the first iteration with

    forrtl: severe (170): Program Exception - stack overflow

Raising the reserve only raises the LIMIT: Windows hands out stack memory as
it is used, so a large value costs nothing on a 64-bit machine.

Usage (from the repository root):
    python Utilities/set_exe_stack.py Solver/Solver.exe            show the current value
    python Utilities/set_exe_stack.py Solver/Solver.exe 1024       set it to 1024 MB

The original file is kept next to it as <name>.exe.bak the first time it is
changed. Only the 8-byte stack-reserve field is modified.
"""
import os
import shutil
import struct
import sys


def _stack_field_offset(data: bytes) -> int:
    if data[:2] != b"MZ":
        raise ValueError("not a Windows executable (no MZ header)")
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    if data[pe:pe + 4] != b"PE\0\0":
        raise ValueError("not a Windows executable (no PE header)")
    opt = pe + 24
    magic = struct.unpack_from("<H", data, opt)[0]
    if magic != 0x20B:
        raise ValueError("only 64-bit (PE32+) executables are supported")
    sec_size = struct.unpack_from("<I", data, opt + 112 + 4 * 8 + 4)[0]
    if sec_size:
        raise ValueError("the executable is digitally signed; changing it would break the signature")
    return opt + 72                                   # SizeOfStackReserve (8 bytes), then SizeOfStackCommit


def stack_reserve(path: str) -> int:
    """Stack reserve of the executable, in bytes."""
    with open(path, "rb") as fh:
        data = fh.read(4096)
    return struct.unpack_from("<Q", data, _stack_field_offset(data))[0]


def set_stack_reserve(path: str, megabytes: float) -> int:
    """Set the stack reserve; returns the previous value in bytes. Keeps <path>.bak once."""
    new = int(megabytes * 1024 * 1024)
    if not 1024 * 1024 <= new <= 16 * 1024 ** 3:
        raise ValueError("stack size must be between 1 MB and 16384 MB")
    with open(path, "rb") as fh:
        data = bytearray(fh.read())
    off = _stack_field_offset(bytes(data[:4096]))
    old = struct.unpack_from("<Q", data, off)[0]
    if old == new:
        return old
    bak = path + ".bak"
    if not os.path.exists(bak):
        shutil.copy2(path, bak)
    struct.pack_into("<Q", data, off, new)
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, path)
    return old


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print(__doc__)
        sys.exit(1)
    exe = sys.argv[1]
    try:
        if len(sys.argv) == 2:
            print(f"{exe}: stack reserve {stack_reserve(exe) / 2 ** 20:g} MB")
        else:
            before = set_stack_reserve(exe, float(sys.argv[2]))
            print(f"{exe}: stack reserve {before / 2 ** 20:g} MB -> {stack_reserve(exe) / 2 ** 20:g} MB "
                  f"(original kept as {os.path.basename(exe)}.bak)")
    except (OSError, ValueError) as e:
        print(f"ERROR: {e}")
        sys.exit(1)