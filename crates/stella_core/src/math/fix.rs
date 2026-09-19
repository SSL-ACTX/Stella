// crates/stella_core/src/math/fix.rs

use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use serde::{Deserialize, Serialize};

pub const FRACTIONAL_BITS: usize = 32;
pub const ONE_RAW: i64 = 1 << FRACTIONAL_BITS;

/// Fixed-Point Number in Q32.32 format.
/// Guarantees bit-perfect determinism across platforms by avoiding IEEE 754 floating point.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default, Hash, Serialize, Deserialize,
)]
pub struct Q32(pub i64);

impl Q32 {
    pub const ZERO: Self = Q32(0);
    pub const ONE: Self = Q32(ONE_RAW);
    pub const NEG_ONE: Self = Q32(-ONE_RAW);
    pub const HALF: Self = Q32(ONE_RAW / 2);

    #[inline(always)]
    pub const fn from_raw(raw: i64) -> Self {
        Self(raw)
    }

    #[inline(always)]
    pub const fn to_raw(self) -> i64 {
        self.0
    }

    #[inline(always)]
    pub const fn from_i64(i: i64) -> Self {
        Self(i << FRACTIONAL_BITS)
    }

    #[inline(always)]
    pub fn from_f64(f: f64) -> Self {
        Self((f * (1i64 << FRACTIONAL_BITS) as f64).round() as i64)
    }

    #[inline(always)]
    pub fn to_f64(self) -> f64 {
        self.0 as f64 / (1i64 << FRACTIONAL_BITS) as f64
    }

    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        if self.0 > other.0 {
            self
        } else {
            other
        }
    }

    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        if self.0 < other.0 {
            self
        } else {
            other
        }
    }

    #[inline(always)]
    pub fn clamp_01(self) -> Self {
        self.max(Self::ZERO).min(Self::ONE)
    }

    #[inline(always)]
    pub fn clamp_01_raw(raw: i64) -> Self {
        if raw <= 0 {
            Self::ZERO
        } else if raw >= ONE_RAW {
            Self::ONE
        } else {
            Self(raw)
        }
    }

    #[inline(always)]
    pub fn mul_fast(self, rhs: Self) -> Self {
        let a = self.0 as i128;
        let b = rhs.0 as i128;
        Self(((a * b) >> FRACTIONAL_BITS) as i64)
    }

    #[inline(always)]
    pub fn abs(self) -> Self {
        Self(self.0.abs())
    }
}

impl Add for Q32 {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(self.0.checked_add(rhs.0).expect("Q32 addition overflow"))
    }
}

impl AddAssign for Q32 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Neg for Q32 {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self {
        Self(-self.0)
    }
}

impl Neg for &Q32 {
    type Output = Q32;

    #[inline(always)]
    fn neg(self) -> Q32 {
        Q32(-self.0)
    }
}

impl Sub for Q32 {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(self.0.checked_sub(rhs.0).expect("Q32 subtraction overflow"))
    }
}

impl SubAssign for Q32 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Mul for Q32 {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        // Cast to i128 to prevent overflow during intermediate multiplication
        let a = self.0 as i128;
        let b = rhs.0 as i128;
        let result = (a * b) >> FRACTIONAL_BITS;
        Self(result.try_into().expect("Q32 multiplication overflow"))
    }
}

impl MulAssign for Q32 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Div for Q32 {
    type Output = Self;

    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        assert_ne!(rhs.0, 0, "Q32 division by zero");
        // Shift left before division to maintain fractional precision
        let a = (self.0 as i128) << FRACTIONAL_BITS;
        let b = rhs.0 as i128;
        Self((a / b).try_into().expect("Q32 division overflow"))
    }
}

impl DivAssign for Q32 {
    #[inline(always)]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

pub const Q16_FRACTIONAL_BITS: usize = 16;
pub const Q16_ONE_RAW: i32 = 1 << Q16_FRACTIONAL_BITS;

/// High-velocity 32-bit Fixed-Point Number in Q16.16 format.
/// Enables 128-bit ARM NEON SIMD vectorization (4 parallel multiply-accumulates per instruction).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default, Hash, Serialize, Deserialize,
)]
pub struct Q16(pub i32);

impl Q16 {
    pub const ZERO: Self = Q16(0);
    pub const ONE: Self = Q16(Q16_ONE_RAW);
    pub const NEG_ONE: Self = Q16(-Q16_ONE_RAW);
    pub const HALF: Self = Q16(Q16_ONE_RAW / 2);

    #[inline(always)]
    pub const fn from_raw(raw: i32) -> Self {
        Self(raw)
    }

    #[inline(always)]
    pub const fn to_raw(self) -> i32 {
        self.0
    }

    #[inline(always)]
    pub const fn from_i32(i: i32) -> Self {
        Self(i << Q16_FRACTIONAL_BITS)
    }

    #[inline(always)]
    pub fn from_f64(f: f64) -> Self {
        Self((f * (1i32 << Q16_FRACTIONAL_BITS) as f64).round() as i32)
    }

    #[inline(always)]
    pub fn to_f64(self) -> f64 {
        self.0 as f64 / (1i32 << Q16_FRACTIONAL_BITS) as f64
    }

    #[inline(always)]
    pub fn from_q32(q: Q32) -> Option<Self> {
        // Shift right by 16 bits to convert Q32.32 -> Q16.16
        let shifted = q.0 >> 16;
        if shifted >= i32::MIN as i64 && shifted <= i32::MAX as i64 {
            Some(Self(shifted as i32))
        } else {
            None
        }
    }

    #[inline(always)]
    pub fn to_q32(self) -> Q32 {
        Q32((self.0 as i64) << 16)
    }

    #[inline(always)]
    pub fn clamp_01_raw(raw: i64) -> Self {
        if raw <= 0 {
            Self::ZERO
        } else if raw >= Q16_ONE_RAW as i64 {
            Self::ONE
        } else {
            Self(raw as i32)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixed_point_precision() {
        // Basic arithmetic constraints
        let a = Q32::from_f64(1.0);
        let b = Q32::from_f64(2.0);
        assert_eq!(a + b, Q32::from_f64(3.0));
        assert_eq!(b - a, Q32::from_f64(1.0));

        // Fractional multiplication
        let c = Q32::from_f64(1.5);
        let d = Q32::from_f64(2.25);
        assert_eq!(c * d, Q32::from_f64(3.375));

        // Fractional division
        assert_eq!(d / c, Q32::from_f64(1.5));

        // Ensure bit-perfect integer bounds
        let e = Q32::from_i64(5);
        let f = Q32::from_i64(5);
        assert_eq!(e * f, Q32::from_i64(25));
    }
}
