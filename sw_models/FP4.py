"""Software model of the 4-bit floating-point format [1,2,1] (FP4 E2M1).

    bit 3 = sign, bits 2..1 = exponent (E), bit 0 = fraction (F)

    normal     (E = 1..3): (-1)^S * 1.F * 2^(E - 1)
    subnormal  (E = 0)   : (-1)^S * 0.F * 2^(1 - 1)      -> only +-0.5 (F = 1)
    zero       (E = 0, F = 0): +0 and -0

    No Inf/NaN: E = 3 is a normal exponent and overflow saturates at +-6.
    Non-negative values: 0, 0.5, 1, 1.5, 2, 3, 4, 6.

Rounding: round-to-nearest, ties-to-even. Every operation computes an EXACT result
(integer significand M times 2^x) and rounds once in `_pack`.

Change E_BITS / F_BITS below to experiment with other tiny formats (e.g. 1, 2 for [1,1,2]).
Run `python fp4_121.py` to see the value table, examples and the self-test.
"""
import itertools
import math
from fractions import Fraction

E_BITS = 2
F_BITS = 1
BIAS = (1 << (E_BITS - 1)) - 1          # 1
EMIN = 1 - BIAS                          # exponent of subnormals and of the smallest normal
WIDTH = 1 + E_BITS + F_BITS
MAX_MAG = (1 << (E_BITS + F_BITS)) - 1   # largest magnitude code (exp and fraction all ones)


class FP4:
    """One number of the format, stored as its bit pattern."""

    __slots__ = ("bits",)

    def __init__(self, bits: int):
        if not 0 <= bits < (1 << WIDTH):
            raise ValueError(f"bits must be in 0..{(1 << WIDTH) - 1}")
        self.bits = bits

    # ---- fields ---------------------------------------------------------
    @property
    def sign(self) -> int:
        return self.bits >> (E_BITS + F_BITS)

    @property
    def exp(self) -> int:
        return (self.bits >> F_BITS) & ((1 << E_BITS) - 1)

    @property
    def frac(self) -> int:
        return self.bits & ((1 << F_BITS) - 1)

    @classmethod
    def from_fields(cls, sign: int, exp: int, frac: int) -> "FP4":
        if not (sign in (0, 1) and 0 <= exp < (1 << E_BITS) and 0 <= frac < (1 << F_BITS)):
            raise ValueError("field out of range")
        return cls((sign << (E_BITS + F_BITS)) | (exp << F_BITS) | frac)

    # ---- exact decode: value = (-1)^s * M * 2^x ----------------------------
    def _decode(self):
        if self.exp == 0:
            return self.sign, self.frac, EMIN - F_BITS
        return self.sign, (1 << F_BITS) | self.frac, self.exp - BIAS - F_BITS

    def is_zero(self) -> bool:
        return self.exp == 0 and self.frac == 0

    def is_subnormal(self) -> bool:
        return self.exp == 0 and self.frac != 0

    # ---- conversion -----------------------------------------------------
    def to_fraction(self) -> Fraction:
        s, M, x = self._decode()
        v = Fraction(M) * Fraction(2) ** x
        return -v if s else v

    def to_float(self) -> float:
        s, M, x = self._decode()
        v = math.ldexp(M, x)
        return -v if s else v

    __float__ = to_float

    @classmethod
    def from_float(cls, val: float) -> "FP4":
        """Round a Python float to the nearest FP4 (ties to even, saturating)."""
        if math.isnan(val) or math.isinf(val):
            raise ValueError("format has no Inf/NaN")
        sign = 1 if math.copysign(1.0, val) < 0 else 0
        if val == 0.0:
            return cls._pack(sign, 0, 0)
        m, e = math.frexp(abs(val))             # abs(val) = m * 2^e, m in [0.5, 1)
        return cls._pack(sign, int(m * (1 << 53)), e - 53)   # m * 2^53 is an exact integer

    @classmethod
    def _pack(cls, sign: int, M: int, x: int) -> "FP4":
        """Round the exact value (-1)^sign * M * 2^x to the format and encode it."""
        if M == 0:
            return cls.from_fields(sign, 0, 0)
        E = x + M.bit_length() - 1              # exponent of the leading one
        q = max(E, EMIN) - F_BITS               # exponent of the last place we keep
        shift = q - x
        if shift <= 0:
            Mq = M << (-shift)                  # exact
        else:
            Mq = M >> shift
            rem = M & ((1 << shift) - 1)
            half = 1 << (shift - 1)
            if rem > half or (rem == half and (Mq & 1)):
                Mq += 1                         # round to nearest, ties to even
        if E >= EMIN:                           # normal; a carry ripples into the exponent
            mag = ((E - EMIN + 1) << F_BITS) + (Mq - (1 << F_BITS))
        else:                                   # subnormal; Mq == 2^F is the smallest normal
            mag = Mq
        mag = min(mag, MAX_MAG)                 # saturate
        return cls((sign << (E_BITS + F_BITS)) | mag)

    # ---- arithmetic -----------------------------------------------------
    def __neg__(self) -> "FP4":
        return FP4(self.bits ^ (1 << (E_BITS + F_BITS)))

    def __abs__(self) -> "FP4":
        return FP4(self.bits & MAX_MAG)

    @staticmethod
    def _add_exact(a, b) -> "FP4":
        """a, b are exact (sign, M, x) triples."""
        s1, M1, x1 = a
        s2, M2, x2 = b
        x = min(x1, x2)                         # align: shift the larger one left, nothing is lost
        v = (-1 if s1 else 1) * (M1 << (x1 - x)) + (-1 if s2 else 1) * (M2 << (x2 - x))
        if v == 0:                              # exact zero: -0 only for (-0) + (-0)
            return FP4._pack(s1 & s2 if (M1 == 0 and M2 == 0) else 0, 0, 0)
        return FP4._pack(1 if v < 0 else 0, abs(v), x)

    @staticmethod
    def _mul_exact(a, b):
        return a[0] ^ b[0], a[1] * b[1], a[2] + b[2]

    def __add__(self, o: "FP4") -> "FP4":
        return FP4._add_exact(self._decode(), o._decode())

    def __sub__(self, o: "FP4") -> "FP4":
        return self + (-o)

    def __mul__(self, o: "FP4") -> "FP4":
        return FP4._pack(*FP4._mul_exact(self._decode(), o._decode()))

    @staticmethod
    def mac(a: "FP4", b: "FP4", c: "FP4", fused: bool = True) -> "FP4":
        """a*b + c. fused: exact product, ONE rounding. not fused: round(round(a*b) + c)."""
        if fused:
            return FP4._add_exact(FP4._mul_exact(a._decode(), b._decode()), c._decode())
        return (a * b) + c

    # ---- display / comparison ----------------------------------------------
    def bitstring(self) -> str:
        return f"{self.sign} {self.exp:0{E_BITS}b} {self.frac:0{F_BITS}b}"

    def __repr__(self) -> str:
        return f"FP4({self.bitstring()} = {self.to_float():g})"

    def __eq__(self, o) -> bool:
        return isinstance(o, FP4) and self.bits == o.bits

    def __hash__(self) -> int:
        return hash(self.bits)


def all_codes():
    return [FP4(b) for b in range(1 << WIDTH)]


# --------------------------------------------------------------------------
# independent self-test: exact rationals + brute-force "nearest value, ties to even"
# --------------------------------------------------------------------------
def _oracle_round(r: Fraction, zero_sign: int) -> int:
    codes = [(FP4(b).to_fraction(), b) for b in range(MAX_MAG + 1)]   # non-negative codes, ascending
    if r == 0:
        return zero_sign << (E_BITS + F_BITS)
    s = 1 if r < 0 else 0
    a = abs(r)
    if a >= codes[-1][0]:
        mag = codes[-1][1]                                            # saturate
    else:
        best = None
        for v, b in codes:
            d = abs(a - v)
            if best is None or d < best[0] or (d == best[0] and b % 2 == 0):
                best = (d, b)
        mag = best[1]
    return (s << (E_BITS + F_BITS)) | mag


def self_test() -> None:
    xs = all_codes()
    val = lambda x: x.to_fraction()
    zs = lambda x: x.sign if x.is_zero() else None
    n = 0
    for a, b in itertools.product(xs, repeat=2):
        both0 = a.is_zero() and b.is_zero()
        assert (a + b).bits == _oracle_round(val(a) + val(b), a.sign & b.sign if both0 else 0), (a, b, "add")
        assert (a - b).bits == _oracle_round(val(a) - val(b), a.sign & (1 - b.sign) if both0 else 0), (a, b, "sub")
        assert (a * b).bits == _oracle_round(val(a) * val(b), a.sign ^ b.sign), (a, b, "mul")
        n += 3
    for a, b, c in itertools.product(xs, repeat=3):
        p = val(a) * val(b)
        exact0 = (a.is_zero() or b.is_zero()) and c.is_zero()
        z = ((a.sign ^ b.sign) & c.sign) if exact0 else 0
        assert FP4.mac(a, b, c, True).bits == _oracle_round(p + val(c), z), (a, b, c, "mac fused")
        pr = a * b
        both0 = pr.is_zero() and c.is_zero()
        assert FP4.mac(a, b, c, False).bits == _oracle_round(val(pr) + val(c), pr.sign & c.sign if both0 else 0), (a, b, c, "mac")
        n += 2
    for x in xs:                                                      # decode / encode round trip
        assert FP4.from_float(x.to_float()).bits == x.bits
    print(f"self-test passed: {n} operations (all ADD, SUB, MUL pairs, all MAC triples, both variants)")


if __name__ == "__main__":
    print(f"Format [1,{E_BITS},{F_BITS}], bias {BIAS}, {1 << WIDTH} codes\n")
    print("code  s e f   value")
    for x in all_codes():
        print(f"{x.bits:04b}  {x.bitstring()}  {x.to_float():g}")
    a, b, c = FP4.from_float(1.5), FP4.from_float(3.0), FP4.from_float(0.5)
    print("\n1.5 + 3.0 =", a + b, "(4.5 is a tie between 4 and 6 -> even code 4)")
    print("1.5 * 3.0 =", a * b, "(4.5 again)")
    print("1.5 - 3.0 =", a - b)
    print("1.5 * 3.0 + 0.5 fused    =", FP4.mac(a, b, c, True))
    print("1.5 * 3.0 + 0.5 non-fused =", FP4.mac(a, b, c, False))
    print("6 + 6 =", FP4.from_float(6.0) + FP4.from_float(6.0), "(saturates at 6)\n")
    self_test()
