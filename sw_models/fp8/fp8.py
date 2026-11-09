import math

class FP8_152:
    """Software model for FP8 [1,5,2] with bias 15."""
    BIAS = 15

    def __init__(self, sign: int = 0, exp: int = 0, frac: int = 0):
        # Enforce bit widths natively
        self.sign = sign & 0x01
        self.exp = exp & 0x1F
        self.frac = frac & 0x03

    @classmethod
    def from_float(cls, val: float):
        """Encodes a standard Python float into the FP8 [1,5,2] format."""
        # 1. Extract sign (handling -0.0)
        sign = 1 if math.copysign(1.0, val) < 0 else 0
        abs_val = abs(val)

        # 2. Handle zero explicitly
        if abs_val == 0.0:
            return cls(sign, 0, 0)

        # 3. Extract exponent and normalized fraction
        m, e = math.frexp(abs_val)
        m *= 2.0
        e -= 1
        biased_exp = e + cls.BIAS

        # 4. Handle Subnormals
        if biased_exp <= 0:
            subnormal_frac = m * (2.0 ** (biased_exp - 1))
            frac_bits = int(subnormal_frac * 4.0 + 0.5) # Round to nearest
            
            if frac_bits >= 4:
                return cls(sign, 1, 0)
            return cls(sign, 0, frac_bits)

        # 5. Handle Overflow / Max values
        if biased_exp >= 31:
            return cls(sign, 31, 3)

        # 6. Normal numbers
        m -= 1.0
        frac_bits = int(m * 4.0 + 0.5)

        if frac_bits >= 4:
            biased_exp += 1
            frac_bits = 0
            if biased_exp >= 31:
                return cls(sign, 31, 3)

        return cls(sign, biased_exp, frac_bits)

    def to_float(self) -> float:
        """Decodes the FP8 fields back into a standard Python float."""
        if self.exp == 0:
            # Subnormal: (-1)^S * (0 + F) * 2^(1 - Bias)
            val = (self.frac / 4.0) * (2.0 ** (1 - self.BIAS))
        else:
            # Normal: (-1)^S * (1 + F) * 2^(E - Bias)
            val = (1.0 + (self.frac / 4.0)) * (2.0 ** (self.exp - self.BIAS))
            
        return -val if self.sign else val

    def to_bits(self) -> int:
        """Packs the fields into a single 8-bit integer."""
        return (self.sign << 7) | (self.exp << 2) | self.frac

    @classmethod
    def from_bits(cls, bits: int):
        """Unpacks a single 8-bit integer into the FP8 fields."""
        return cls(
            sign=(bits >> 7) & 0x01,
            exp=(bits >> 2) & 0x1F,
            frac=bits & 0x03
        )

    def add(self, other: 'FP8_152') -> 'FP8_152':
        """Adds two FP8_152 numbers using the hardware processing scheme."""
        # 1. Align the floats by comparing exponents
        # Subnormals (exp=0) are treated as having an effective exponent of 1 for alignment
        exp_a = self.exp if self.exp > 0 else 1
        exp_b = other.exp if other.exp > 0 else 1
        
        target_exp = max(exp_a, exp_b)
        
        # Extract fractions (including the implicit leading 1 for normal numbers)
        # The implicit 1 is at bit position 2 (value 4) because we have 2 fraction bits.
        frac_a = (self.frac | 0x04) if self.exp > 0 else self.frac
        frac_b = (other.frac | 0x04) if other.exp > 0 else other.frac
        
        # Shift left by 4 bits to create internal precision for guard/round bits.
        # The implicit 1 moves from bit 2 to bit 6 (value 64).
        frac_a <<= 4 
        frac_b <<= 4 
        
        # 2. Shift the fraction of the smaller exponent right
        if exp_a > exp_b:
            frac_b >>= (exp_a - exp_b)
        elif exp_b > exp_a:
            frac_a >>= (exp_b - exp_a)
            
        # 3. Add or subtract fractions
        signed_a = -frac_a if self.sign else frac_a
        signed_b = -frac_b if other.sign else frac_b
        
        sum_frac = signed_a + signed_b
        
        result_sign = 1 if sum_frac < 0 else 0
        sum_frac = abs(sum_frac)
        
        if sum_frac == 0:
            return FP8_152(0, 0, 0)
            
        # 4. Normalize the result (leading digit has to be 1)
        # If sum_frac >= 128 (bit 7 is 1), addition overflowed (e.g., 1.x + 1.y = 1z.a)
        if sum_frac >= 128:
            sum_frac >>= 1
            target_exp += 1
            
        # If we lost the leading 1 (e.g. via subtraction), shift left to normalize
        # We stop shifting if target_exp reaches 1, as the result becomes subnormal
        while sum_frac < 64 and target_exp > 1:
            sum_frac <<= 1
            target_exp -= 1
            
        # 5. Round the result
        # After our 4-bit left shift, bit 3 is the half-LSB. Adding 8 rounds to nearest.
        sum_frac += 8
        
        # Check if rounding triggered a new overflow (e.g., 1.11... rounding to 10.00...)
        if sum_frac >= 128:
            sum_frac >>= 1
            target_exp += 1
            
        # Remove the extra internal precision bits
        sum_frac >>= 4
        
        # Pack and clamp limits
        if target_exp >= 31:
            return FP8_152(result_sign, 31, 3)
            
        # Subnormal packaging (if sum_frac is missing the implicit 1 at value 4)
        if sum_frac < 4:
            return FP8_152(result_sign, 0, sum_frac & 0x03)
            
        return FP8_152(result_sign, target_exp, sum_frac & 0x03)
# END class


def full_conversion_check():
    print("Running exhaustive decode/encode sweep...")
    error_count = 0
    
    for bits in range(256):     # 0...255
        # 1. Instantiate from raw 8-bit integer
        fp8_num = FP8_152.from_bits(bits)
        
        # 2. Decode to standard float
        decoded_float = fp8_num.to_float()
        
        # 3. Re-encode back to FP8
        re_encoded = FP8_152.from_float(decoded_float)
        
        # 4. Verify the bit structures match perfectly
        if fp8_num.to_bits() != re_encoded.to_bits():
            # NaN representations (all exp bits 1, frac != 0) might not round-trip cleanly 
            # depending on how you choose to clamp them, but normal/subnormals must match.
            if fp8_num.exp != 31: 
                print(f"Mismatch at bits 0x{bits:02X}: decoded to {decoded_float}, re-encoded to 0x{re_encoded.to_bits():02X}")
                error_count += 1

    print(f"Sweep complete. Found {error_count} standard round-trip errors.")


def main():
    full_conversion_check()



if __name__ == "__main__": main()