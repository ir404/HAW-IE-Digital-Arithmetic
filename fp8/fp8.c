#include <stdio.h>
#include <stdint.h>
#include <math.h>

#define FP8_BIAS 15

typedef struct {
    uint8_t frac : 2;       // LSB
    uint8_t exp  : 5;
    uint8_t sign : 1;       // MSB
} FP8;

FP8 double_to_fp8(double val) {
    FP8 result;
    
    // 1. Extract sign
    result.sign = (val < 0.0 || (val == 0.0 && signbit(val))) ? 1 : 0;
    
    double abs_val = fabs(val);
    
    // 2. Handle zero explicitly
    if (abs_val == 0.0) {
        result.exp = 0;
        result.frac = 0;
        return result;
    }
    
    // 3. Extract exponent and normalized fraction using frexp
    // frexp returns fraction in [0.5, 1.0) and exponent such that val = fraction * 2^exponent
    int c_exp;
    double fraction = frexp(abs_val, &c_exp);
    
    // Adjust to standard normalized form (1.f * 2^e)
    // We want the fraction to be in [1.0, 2.0)
    fraction *= 2.0;
    c_exp -= 1;
    
    // Calculate the biased exponent
    int biased_exp = c_exp + FP8_BIAS;
    
    // 4. Handle Subnormals (biased_exp <= 0)
    if (biased_exp <= 0) {
        result.exp = 0;
        // Shift fraction based on how far below normal we are
        // A bias of 15 means normal exponents start at -14 (biased_exp = 1)
        // Subnormals effectively have an exponent of -14, but no leading 1
        // The scale factor is 2^(biased_exp - 1)
        double subnormal_frac = fraction * pow(2.0, biased_exp - 1);
        
        // Extract 2 fraction bits, adding 0.5 for rounding to nearest
        int frac_bits = (int)(subnormal_frac * 4.0 + 0.5);
        
        // Handle rounding overflow back into normal range
        if (frac_bits >= 4) {
            result.exp = 1;
            result.frac = 0;
        } else {
            result.frac = frac_bits;
        }
        return result;
    }
    
    // 5. Handle Overflow / Infinity / Max values
    if (biased_exp >= 31) {
        // Clamp to max value or represent infinity if defined.
        // Assuming standard clamp to max for now.
        result.exp = 31;
        result.frac = 3; // 1.11_2
        return result;
    } 
    
    // 6. Normal numbers
    result.exp = biased_exp;
    
    // Extract the fraction bits (remove leading 1.0)
    fraction -= 1.0;
    
    // Multiply by 4 (since we have 2 fraction bits) and round
    int frac_bits = (int)(fraction * 4.0 + 0.5);
    
    // Handle rounding overflow
    if (frac_bits >= 4) {
        result.exp += 1;
        result.frac = 0;
        // Check overflow again after rounding up
        if (result.exp >= 31) {
            result.exp = 31;
            result.frac = 3;
        }
    } else {
        result.frac = frac_bits;
    }
    
    return result;
}

void test_double_to_fp8() {

}

int main(void) {

}
