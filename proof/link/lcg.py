"""Exact java.util.Random (JDK 6/8, identical algorithm) LCG stepping with O(log N) jumps.

State = the 48-bit `seed` field inside java.util.Random (after scrambling).
One next(bits) call advances the state by exactly one LCG step.
Math.random() in JDK 6 = nextDouble() on a lazily created static `new Random()`:
two LCG steps per call (next(26), next(27)).

Validated against Java 8 (proof/jdk) by validate_lcg.py -> lcg-validation.json.
"""
A = 0x5DEECE66D
C = 0xB
M = 1 << 48
MASK = M - 1
A_INV = pow(A, -1, M)          # exists because A is odd


def affine(steps):
    """(mul, add) such that advance(s, steps) = (mul*s + add) mod 2^48. Any integer steps (negative = backward)."""
    steps %= M                 # full period 2^48 (Hull-Dobell: C odd, A-1 divisible by 4)
    mul, add, a, c = 1, 0, A, C
    while steps:
        if steps & 1:
            mul, add = (a * mul) % M, (a * add + c) % M
        c = (c * (a + 1)) % M
        a = (a * a) % M
        steps >>= 1
    return mul, add


def advance(state, steps):
    mul, add = affine(steps)
    return (mul * state + add) & MASK


def back(state, steps):
    """Independent backward implementation using the inverse map s = A^-1 (s' - C); used to cross-check advance(-n)."""
    mul, add = 1, 0                                  # compose inverse map 'steps' times by doubling
    a, c = A_INV, (-A_INV * C) % M
    n = steps
    while n:
        if n & 1:
            mul, add = (a * mul) % M, (a * add + c) % M
        c = (c * (a + 1)) % M
        a = (a * a) % M
        n >>= 1
    return (mul * state + add) & MASK


def scramble(seed):
    """new Random(seed) sets internal state to (seed ^ A) & MASK."""
    return (seed ^ A) & MASK


def unscramble(state):
    """Low 48 bits of the constructor seed that produced `state` (high 16 bits are lost)."""
    return (state ^ A) & MASK


def _signed32(v):
    v &= 0xFFFFFFFF
    return v - (1 << 32) if v & 0x80000000 else v


class JRandom:
    """Bit-exact java.util.Random. `state` is the internal 48-bit seed."""

    def __init__(self, seed=None, state=None):
        self.state = scramble(seed) if state is None else state & MASK
        self.calls = 0             # LCG steps consumed

    def next(self, bits):
        self.state = (self.state * A + C) & MASK
        self.calls += 1
        return _signed32(self.state >> (48 - bits)) if bits == 32 else self.state >> (48 - bits)

    def next_int(self, bound=None):
        if bound is None:
            return self.next(32)
        if bound <= 0:
            raise ValueError
        if bound & -bound == bound:
            return (bound * self.next(31)) >> 31
        while True:
            bits = self.next(31)
            val = bits % bound
            if bits - val + (bound - 1) < (1 << 31):
                return val

    def next_long(self):
        hi = self.next(32)
        lo = self.next(32)
        v = ((hi << 32) + lo) & 0xFFFFFFFFFFFFFFFF
        return v - (1 << 64) if v >> 63 else v

    def next_boolean(self):
        return self.next(1) != 0

    def next_float(self):
        return self.next(24) / float(1 << 24)

    def next_double(self):
        return ((self.next(26) << 27) + self.next(27)) * (1.0 / (1 << 53))


def to_float32(x):
    import struct
    return struct.unpack('f', struct.pack('f', x))[0]


def reset_xz(state_before, width=256, height=256):
    """Player.resetPos: x=(float)Math.random()*width, z=(float)Math.random()*height. Returns (x, z, state_after)."""
    r = JRandom(state=state_before)
    x = to_float32(to_float32(r.next_double()) * width)
    z = to_float32(to_float32(r.next_double()) * height)
    return x, z, r.state
