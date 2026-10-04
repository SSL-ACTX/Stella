// crates/stella_compiler/src/chaos.rs

pub struct HenonMap {
    x: f64,
    y: f64,
    a: f64,
    b: f64,
}

impl HenonMap {
    /// Initializes a new Hénon Map with classical chaotic parameters (a=1.4, b=0.3).
    /// Projects arbitrary seeds safely into the basin of attraction and runs a warm-up phase.
    pub fn new(seed_x: f64, seed_y: f64) -> Self {
        // Project arbitrary seed into the safe basin near the attractor
        let x_norm = ((seed_x.abs() * 1000.0).fract() * 0.8) - 0.4;
        let y_norm = ((seed_y.abs() * 1000.0).fract() * 0.4) - 0.2;

        let mut map = Self {
            x: if x_norm.is_nan() { 0.1 } else { x_norm },
            y: if y_norm.is_nan() { 0.1 } else { y_norm },
            a: 1.4,
            b: 0.3,
        };

        // Warm-up phase: discard the initial transient trajectory
        for _ in 0..1_000 {
            map.step();
        }

        map
    }

    /// Advances the chaotic system by one discrete time step with attractor containment.
    fn step(&mut self) {
        let next_x = 1.0 - self.a * self.x * self.x + self.y;
        let next_y = self.b * self.x;

        if next_x.is_nan() || next_x.abs() > 2.0 || next_y.is_nan() || next_y.abs() > 2.0 {
            self.x = ((self.x.abs() * 17.0).fract() * 0.8) - 0.4;
            self.y = ((self.y.abs() * 13.0).fract() * 0.4) - 0.2;
        } else {
            self.x = next_x;
            self.y = next_y;
        }
    }

    /// Extracts entropy from the chaotic state to generate a pseudo-random usize.
    pub fn next_usize(&mut self) -> usize {
        self.step();
        let entropy = (self.x.abs() * 1_000_000_000.0).fract() * 1_000_000_000.0;
        entropy as usize
    }
}

/// Time-varying chaotic coordinate manifold for continuous terminal rolling.
/// Produces a deterministic, high-dimensional non-stationary orbit zeta(t) in [0.0, 1.0)
/// parameterized by chaotic attractor seeds.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ChaoticManifold {
    pub seed_x: f64,
    pub seed_y: f64,
}

impl ChaoticManifold {
    pub fn new(seed_x: f64, seed_y: f64) -> Self {
        Self { seed_x, seed_y }
    }

    /// Computes the chaotic masking vector zeta(t) for N dimensions at discrete cycle t.
    pub fn mask_at(&self, step: usize, dim: usize) -> alloc::vec::Vec<stella_core::math::Q32> {
        let mut masks = alloc::vec::Vec::with_capacity(dim);
        for d in 0..dim {
            let omega1 = 0.6180339887 * (d + 1) as f64;
            let omega2 = 1.4142135623 * (d * 3 + 7) as f64;
            let phase1 = libm::sin(self.seed_x * 100.0 + (step as f64) * omega1);
            let phase2 = libm::cos(self.seed_y * 100.0 + (step as f64) * omega2);
            let val = libm::fabs(libm::fabs(phase1 * 0.5 + phase2 * 0.5) * 0.999 + 0.0001);
            let val_fract = val - libm::floor(val);
            masks.push(stella_core::math::Q32::from_f64(val_fract));
        }
        masks
    }

    pub fn mask_at_f64(&self, step: usize, dim: usize) -> alloc::vec::Vec<f64> {
        let masks = self.mask_at(step, dim);
        let mut res = alloc::vec::Vec::with_capacity(dim);
        for q in masks {
            res.push(q.to_f64());
        }
        res
    }
}
