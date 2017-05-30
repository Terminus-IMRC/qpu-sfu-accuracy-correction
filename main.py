#!/usr/bin/env python3

import struct
import numpy as np
from videocore.assembler import qpu
from videocore.driver import Driver

def run_code(code, X):
    with Driver() as drv:
        X = drv.copy(np.array(X, dtype = 'float32'))
        Y = drv.alloc((16), dtype = 'float32')
        drv.execute(
                n_threads = 1,
                program = drv.program(boilerplate, code),
                uniforms = [X.address, Y.address])
        return np.copy(Y)

def float_as_uint(f):
    return struct.unpack('!I', struct.pack('!f', f))[0]

def uint_as_float(i):
    return struct.unpack('!f', struct.pack('!I', i))[0]

def print_uint(prefix, v):
    print(prefix, list(map(lambda x: "%08x" % x, v)))

def print_float(prefix, v):
    print(prefix, list(map(lambda x: "%.4e" % x, v)))

def print_float_as_uint(prefix, v):
    print_uint(prefix, list(map(float_as_uint, v)))

@qpu
def boilerplate(asm, f):
    setup_dma_load(nrows = 1)
    start_dma_load(uniform)
    wait_dma_load()
    setup_vpm_read(nrows = 1)
    setup_vpm_write()

    f(asm)

    setup_dma_store(nrows = 1)
    start_dma_store(uniform)
    wait_dma_store()
    exit()

@qpu
def code_sfu_recip_0(asm):
    mov(r0, vpm)
    mov(sfu_recip, r0)
    nop()
    nop()
    mov(vpm, r4)

@qpu
def code_sfu_recip_1(asm):
    mov(r0, vpm)
    mov(sfu_recip, r0)
    nop()
    nop()
    fmul(r0, r0, r4)
    fsub(r0, 2.0, r0)
    fmul(vpm, r4, r0)

@qpu
def code_sfu_recip_2(asm):
    mov(r0, vpm)
    mov(sfu_recip, r0)
    nop()
    nop()
    # 1
    fmul(r1, r0, r4)
    fsub(r1, 2.0, r1)
    fmul(r1, r4, r1)
    # 2
    fmul(r2, r0, r1)
    fsub(r2, 2.0, r2)
    fmul(vpm, r1, r2)

@qpu
def code_sfu_recip_3(asm):
    mov(r0, vpm)
    mov(sfu_recip, r0)
    nop()
    nop()
    # 1
    fmul(r1, r0, r4)
    fsub(r1, 2.0, r1)
    fmul(r1, r4, r1)
    # 2
    fmul(r2, r0, r1)
    fsub(r2, 2.0, r2)
    fmul(r1, r1, r2)
    # 3
    fmul(r2, r0, r1)
    fsub(r2, 2.0, r2)
    fmul(vpm, r1, r2)

def do_recip(xf):
    print('# recip')

    yf0 = run_code(code_sfu_recip_0, xf)
    yf1 = run_code(code_sfu_recip_1, xf)
    yf2 = run_code(code_sfu_recip_2, xf)
    yf3 = run_code(code_sfu_recip_3, xf)
    yf_ref = map(lambda x: np.float32(1.0) / np.float32(x), xf)

    print_float_as_uint('output 0:  ', yf0)
    print_float_as_uint('output 1:  ', yf1)
    print_float_as_uint('output 2:  ', yf2)
    print_float_as_uint('output 3:  ', yf3)
    print_float_as_uint('output ref:', yf_ref)

def main():
    xi = np.random.uniform(1 << 23, (255 << 23), 16).astype(np.uint32)
    xf = list(map(uint_as_float, xi))
    print_uint('input:', xi)
    print_float('input as float:', xf)
    print()

    do_recip(xf)
    print()

if __name__ == '__main__':
    main()
