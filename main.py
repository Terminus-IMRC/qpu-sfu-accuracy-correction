#!/usr/bin/env python3

import sys
import struct
import numpy as np
from videocore.assembler import qpu
from videocore.driver import Driver

def run_code(code, X, n):
    with Driver() as drv:
        X = drv.copy(np.array(X, dtype = np.float32))
        Y = drv.alloc((n, 16), dtype = np.float32)
        drv.execute(
                n_threads = 1,
                program = drv.program(boilerplate, code, n),
                uniforms = [X.address, Y.address],
                timeout = 5)
        return np.copy(Y)

def float_as_uint(f):
    return np.vectorize(
            lambda x: struct.unpack('!I', struct.pack('!f', x))[0])(f)

def uint_as_float(i):
    return np.vectorize(
            lambda x: struct.unpack('!f', struct.pack('!I', x))[0])(i)

def pretty_uint(v):
    return ' '.join([f'{x:08x}' for x in v])

def pretty_float(v):
    return ' '.join([f'{x:.2e}' for x in v])

@qpu
def boilerplate(asm, f, n):
    setup_dma_load(nrows = 1)
    start_dma_load(uniform)
    wait_dma_load()
    setup_vpm_read(nrows = 1)
    setup_vpm_write()

    f(asm, n)

    setup_dma_store(nrows = n)
    start_dma_store(uniform)
    wait_dma_store()
    exit()

@qpu
def code_sfu_recip(asm, n):
    mov(r0, vpm)
    mov(sfu_recip, r0)
    nop()
    nop()
    mov(r1, r4).mov(vpm, r4)
    # r0 = a, r1 = x_n
    for i in range(n):
        fmul(r2, r0, r1)
        fsub(r2, 2.0, r2)
        fmul(r1, r1, r2)
        mov(vpm, r1)

@qpu
def code_sfu_recip_improved(asm, n):
    mov(r0, vpm)
    mov(sfu_recip, r0)
    nop()
    nop()
    mov(r1, r4).mov(vpm, r4)
    # r0 = a, r1 = x_n
    for i in range(n):
        fmul(r2, r0, r1)
        fsub(r2, 1.0, r2)
        fmul(r2, r1, r2)
        fadd(r1, r1, r2)
        mov(vpm, r1)

@qpu
def code_sfu_recipsqrt(asm, n):
    mov(r0, vpm)
    mov(sfu_recipsqrt, r0)
    nop()
    nop()
    mov(r1, r4).mov(vpm, r4)
    # r0 = a, r1 = x_n
    for i in range(n):
        mov(r2, 1.0).fmul(r3, r1, r1)
        fadd(r2, r2, 2.0).fmul(r3, r0, r3)
        fsub(r2, r2, r3).fmul(r3, r1, 0.5)
        fmul(r1, r2, r3)
        mov(vpm, r1)

@qpu
def code_sfu_recipsqrt_improved(asm, n):
    mov(r0, vpm)
    mov(sfu_recipsqrt, r0)
    nop()
    nop()
    mov(r1, r4).mov(vpm, r4)
    # r0 = a, r1 = x_n
    for i in range(n):
        fmul(r2, r1, r1)
        mov(r3, 0.5).fmul(r2, r0, r2)
        fsub(r2, 1.0, r2).fmul(r3, r1, r3)
        fmul(r2, r2, r3)
        fadd(r1, r1, r2)
        mov(vpm, r1)

def do_code_sfu(code_sfu, cpu_func, xf, n):
    print()
    print(f'# {code_sfu.__name__}')

    yf = run_code(code_sfu, xf, n)

    for i in range(n):
        print(f'output {i}:  ', pretty_uint(float_as_uint(yf[i])))

    print('output ref:', pretty_uint(float_as_uint(cpu_func(xf))))

def main():
    np.random.seed(int(sys.argv[1]))
    xi = np.random.uniform(1 << 23, (255 << 23), 16).astype(np.uint32)
    xf = uint_as_float(xi)
    print('input:     ', pretty_float(xf))
    print('input:     ', pretty_uint(xi))

    for code_sfu in code_sfu_recip, code_sfu_recip_improved:
        do_code_sfu(code_sfu, np.reciprocal, xf, 5)

    for code_sfu in code_sfu_recipsqrt, code_sfu_recipsqrt_improved:
        do_code_sfu(code_sfu, lambda x: np.reciprocal(np.sqrt(x)), xf, 5)

if __name__ == '__main__':
    main()
