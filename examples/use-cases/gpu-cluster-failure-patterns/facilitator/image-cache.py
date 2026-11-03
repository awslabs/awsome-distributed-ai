#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Read the lab's container images into the page cache, or check that they are there.

A container start reads its image from the root volume unless the image's pages
are already in the page cache. On a g7.48xlarge node whose images had not been
read since its last boot, creating the torch container took 168 s of the
launcher's 180 s limit; with the images cached it took about 5 s.

`read` reads every byte of each image once and then checks it. `check` reads no
data: it maps each image read-only and asks mincore(2) which pages are resident.
Both print one line per image and exit 1 unless every page of every image is in
the page cache.
"""
import argparse
import ctypes
import os
import sys
import time

IMAGES = ('/opt/aim344/aim344.sqsh', '/opt/aim344/nccl-baseline.sqsh')
CHUNK = 8 << 20
PROT_READ = 1
MAP_SHARED = 1

libc = ctypes.CDLL(None, use_errno=True)
libc.mmap.restype = ctypes.c_void_p
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
                      ctypes.c_int, ctypes.c_long]
libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]


def read_image(path):
    """Read the whole file once, front to back; return the bytes read."""
    total = 0
    buffer = bytearray(CHUNK)
    with open(path, 'rb', buffering=0) as image:
        os.posix_fadvise(image.fileno(), 0, 0, os.POSIX_FADV_SEQUENTIAL)
        while True:
            count = image.readinto(buffer)
            if not count:
                return total
            total += count


def resident_pages(path):
    """Return (pages, resident pages) of the file, from mincore on a read-only mapping."""
    page = os.sysconf('SC_PAGE_SIZE')
    fd = os.open(path, os.O_RDONLY)
    try:
        size = os.fstat(fd).st_size
        if size == 0:
            raise OSError(f'{path} is empty')
        address = libc.mmap(None, size, PROT_READ, MAP_SHARED, fd, 0)
        if address in (None, ctypes.c_void_p(-1).value):
            raise OSError(ctypes.get_errno(), f'mmap of {path} failed')
        try:
            pages = (size + page - 1) // page
            vector = (ctypes.c_ubyte * pages)()
            if libc.mincore(ctypes.c_void_p(address), ctypes.c_size_t(size), vector) != 0:
                raise OSError(ctypes.get_errno(), f'mincore of {path} failed')
            return pages, sum(byte & 1 for byte in bytes(vector))
        finally:
            libc.munmap(ctypes.c_void_p(address), ctypes.c_size_t(size))
    finally:
        os.close(fd)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('action', choices=('read', 'check'))
    parser.add_argument('images', nargs='*', default=list(IMAGES),
                        help='image files (default: %(default)s)')
    args = parser.parse_args(argv)
    complete = True
    for path in args.images:
        if args.action == 'read':
            start = time.monotonic()
            total = read_image(path)
            seconds = time.monotonic() - start
            print(f'image-cache: {path} read {total} B in {seconds:.1f} s', flush=True)
        pages, resident = resident_pages(path)
        complete = complete and resident == pages
        # Rounded down, so that an image short of even one page never shows 100.0 %.
        print(f'image-cache: {path} resident {resident} of {pages} pages '
              f'({resident * 1000 // pages / 10:.1f} %)', flush=True)
    if complete:
        print('image-cache: every page of every image is in the page cache')
        return 0
    print('image-cache: not every page is in the page cache; run image-cache.py read on this node')
    return 1


if __name__ == '__main__':
    sys.exit(main())
