use crate::layer::dense::{Dense, DenseQ16};
use crate::math::fix::{Q16, Q32};
use crate::math::matrix::Matrix;
use alloc::vec;
use alloc::vec::Vec;

/// The Stella Virtual Machine Runtime.
/// Executes the continuous-state neural representation of discrete assembly logic.
/// Learning rule type for dynamic plasticity
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize, Default)]
pub enum PlasticityKind {
    /// Standard Hebbian correlation: dW = rate * (post * pre - decay * W)
    #[default]
    Hebbian,
    /// Anti-Hebbian correlation: dW = rate * (-post * pre - decay * W)
    AntiHebbian,
    /// Oja's normalized learning rule: dW = rate * (post * pre - post * post * W)
    Oja,
}

/// Dynamic synaptic plasticity rule: updates W[dest, src] based on presynaptic and postsynaptic correlations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PlasticityRule {
    pub pre: usize,
    pub dest: usize,
    pub rate: Q32,
    pub decay: Q32,
    #[serde(default)]
    pub kind: PlasticityKind,
}

/// The Stella Virtual Machine Runtime.
/// Executes the continuous-state neural representation of discrete assembly logic.
pub struct Vm {
    /// The memory of the machine, represented as a single continuous state vector.
    pub state: Matrix,
    /// The compiled logic of the program, represented as a neural layer.
    pub logic_core: Dense,
    /// Dynamic plasticity rules for online continuous learning and trace adaptation.
    pub plasticity_rules: Vec<PlasticityRule>,
    /// Double-buffering scratch vector to guarantee ZERO heap allocations during execution.
    scratch: Vec<Q32>,
    /// High-velocity 128-bit ARM NEON SIMD accelerated core (when weights fit in Q16.16)
    pub simd_core: Option<DenseQ16>,
    simd_state: Vec<Q16>,
    simd_scratch: Vec<Q16>,
    /// Pad neuron slots to skip during convergence check (obfuscation ghost neurons).
    pub pad_mask: Vec<bool>,
}

impl Vm {
    /// Initializes the VM with a compiled logic core and an empty state vector.
    /// Automatically detects if weights fit into Q16.16 to enable the 128-bit NEON SIMD engine.
    pub fn new(state_size: usize, logic_core: Dense) -> Self {
        assert_eq!(
            state_size, logic_core.weights.cols,
            "State size must match logic core input dimensions"
        );
        assert_eq!(
            state_size, logic_core.weights.rows,
            "Logic core must map state back to same dimensions to loop"
        );

        let simd_core = DenseQ16::try_from_dense(&logic_core);

        Self {
            state: Matrix::zeros(state_size, 1),
            logic_core,
            plasticity_rules: Vec::new(),
            scratch: vec![Q32::ZERO; state_size],
            simd_core,
            simd_state: vec![Q16::ZERO; state_size],
            simd_scratch: vec![Q16::ZERO; state_size],
            pad_mask: Vec::new(),
        }
    }

    /// Initializes the VM with plasticity adaptation rules.
    pub fn with_plasticity(
        state_size: usize,
        logic_core: Dense,
        plasticity_rules: Vec<PlasticityRule>,
    ) -> Self {
        let mut vm = Self::new(state_size, logic_core);
        vm.plasticity_rules = plasticity_rules;
        vm
    }

    /// Adds a dynamic plasticity rule to the runtime.
    pub fn add_plasticity_rule(&mut self, rule: PlasticityRule) {
        self.plasticity_rules.push(rule);
    }

    /// Injects external data into the I/O mapped region of the state vector (the first N neurons).
    pub fn write_io(&mut self, inputs: &[Q32]) {
        assert!(
            inputs.len() <= self.state.rows,
            "I/O payload exceeds state vector capacity"
        );
        for (i, &val) in inputs.iter().enumerate() {
            self.state.set(i, 0, val);
        }
    }

    /// Reads the output from the I/O mapped region of the state vector.
    pub fn read_io(&self, count: usize) -> Vec<Q32> {
        assert!(
            count <= self.state.rows,
            "Requested I/O read exceeds state vector capacity"
        );
        let mut outputs = Vec::with_capacity(count);
        for i in 0..count {
            outputs.push(self.state.get(i, 0));
        }
        outputs
    }

    /// Returns the raw state slice.
    #[inline(always)]
    pub fn state_slice(&self) -> &[Q32] {
        &self.state.data
    }

    /// Executes a single clock cycle with ZERO heap allocations.
    /// Mathematically: S_new = Clamp01(W * S_old + B)
    #[inline]
    pub fn step(&mut self) {
        self.logic_core
            .forward_into(&self.state.data, &mut self.scratch);
        core::mem::swap(&mut self.state.data, &mut self.scratch);
    }

    /// Executes a single clock cycle with Hebbian plasticity adaptation:
    /// 1. Forward pass: S_new = Clamp01(W * S_old + B)
    /// 2. Hebbian update: W[dest, src] += rate * (post * pre - decay * W[dest, src])
    #[inline]
    pub fn step_plastic(&mut self) -> bool {
        let changed = self.step_stable();
        self.adapt_plasticity();
        changed
    }

    /// In-place Hebbian weight adaptation across registered plasticity rules.
    #[inline]
    pub fn adapt_plasticity(&mut self) {
        if self.plasticity_rules.is_empty() {
            return;
        }

        let state_data = &self.state.data;
        for rule in &self.plasticity_rules {
            let pre_val = state_data[rule.pre];
            let post_val = state_data[rule.dest];
            let cur_w = self.logic_core.weights.get(rule.dest, rule.pre);

            let delta = match rule.kind {
                PlasticityKind::Hebbian => {
                    // Standard Hebbian: delta = rate * (post * pre - decay * w)
                    let hebb_term = post_val * pre_val;
                    let decay_term = rule.decay * cur_w;
                    rule.rate * (hebb_term - decay_term)
                }
                PlasticityKind::AntiHebbian => {
                    // Anti-Hebbian: delta = rate * (-post * pre - decay * w)
                    let anti_term = -(post_val * pre_val);
                    let decay_term = rule.decay * cur_w;
                    rule.rate * (anti_term - decay_term)
                }
                PlasticityKind::Oja => {
                    // Oja's rule: delta = rate * (post * pre - post * post * w)
                    let hebb_term = post_val * pre_val;
                    let oja_decay = (post_val * post_val) * cur_w;
                    rule.rate * (hebb_term - oja_decay)
                }
            };

            let new_w = (cur_w + delta).clamp_01();
            self.logic_core.weights.set(rule.dest, rule.pre, new_w);
        }
    }

    /// Single clock cycle. Returns `true` if state changed, `false` if converged.
    #[inline]
    pub fn step_stable(&mut self) -> bool {
        self.logic_core
            .forward_into(&self.state.data, &mut self.scratch);

        let converged = if self.pad_mask.is_empty() {
            self.state.data == self.scratch
        } else {
            self.state
                .data
                .iter()
                .zip(self.scratch.iter())
                .enumerate()
                .all(|(i, (old, new))| self.pad_mask.get(i).copied().unwrap_or(false) || old == new)
        };

        if converged {
            false
        } else {
            core::mem::swap(&mut self.state.data, &mut self.scratch);
            true
        }
    }

    /// Propagates the system continuously for a designated number of clock cycles.
    /// Automatically utilizes the 128-bit ARM NEON SIMD engine when available!
    #[inline]
    pub fn run(&mut self, cycles: usize) {
        if cycles == 0 {
            return;
        }

        // 1. FAST-PATH: 128-bit ARM NEON SIMD Execution
        if let Some(ref simd) = self.simd_core {
            // Load state into Q16 buffer
            for (i, val) in self.state.data.iter().enumerate() {
                self.simd_state[i] = Q16::from_q32(*val).unwrap_or(Q16::ZERO);
            }

            let mut pairs = cycles / 2;
            let remainder = cycles % 2;

            let s = &mut self.simd_state;
            let sc = &mut self.simd_scratch;

            while pairs > 0 {
                simd.forward_into(s, sc);
                simd.forward_into(sc, s);
                pairs -= 1;
            }

            if remainder > 0 {
                simd.forward_into(s, sc);
                core::mem::swap(s, sc);
            }

            // Sync back to Q32 state vector
            for (i, &val) in self.simd_state.iter().enumerate() {
                self.state.data[i] = val.to_q32();
            }
            return;
        }

        // 2. FALLBACK: Exact Q32.32 AArch64 Hardware Kernel
        let mut pairs = cycles / 2;
        let remainder = cycles % 2;

        let s = &mut self.state.data;
        let sc = &mut self.scratch;

        while pairs > 0 {
            // Forward pass 1: S -> Scratch
            self.logic_core.forward_into(s, sc);
            // Forward pass 2: Scratch -> S
            self.logic_core.forward_into(sc, s);
            pairs -= 1;
        }

        if remainder > 0 {
            self.logic_core.forward_into(s, sc);
            core::mem::swap(s, sc);
        }
    }

    /// Propagates the system with plasticity for a designated number of clock cycles.
    pub fn run_plastic(&mut self, cycles: usize) {
        for _ in 0..cycles {
            self.step_plastic();
        }
    }

    /// Propagates the system until the state vector converges to a fixed-point attractor
    /// (S_{t+1} == S_t) or until `max_cycles` is reached.
    /// Returns `(cycles_executed, converged)`.
    pub fn run_until_stable(&mut self, max_cycles: usize) -> (usize, bool) {
        for cycle in 0..max_cycles {
            if !self.step_stable() {
                return (cycle + 1, true);
            }
        }
        (max_cycles, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn test_vm_execution_loop() {
        // Build a mock 2-neuron program.
        // Neuron 0: Oscillates (NOT gate of itself).
        // Neuron 1: Copies Neuron 0 (Identity mapping of Neuron 0).
        let w = Matrix::from_vec(
            2,
            2,
            vec![
                Q32::from_f64(-1.0),
                Q32::ZERO,
                Q32::from_f64(1.0),
                Q32::ZERO,
            ],
        );

        let b = Matrix::from_vec(2, 1, vec![Q32::from_f64(1.0), Q32::ZERO]);

        let logic_core = Dense::new(w, b);
        let mut vm = Vm::new(2, logic_core);

        // Initial State: [0.0, 0.0]
        assert_eq!(vm.state.get(0, 0), Q32::ZERO);
        assert_eq!(vm.state.get(1, 0), Q32::ZERO);

        // Cycle 1:
        // N0 = clamp(-1 * 0 + 1) = 1.0
        // N1 = clamp(1 * 0 + 0) = 0.0
        vm.step();
        assert_eq!(vm.state.get(0, 0), Q32::from_f64(1.0));
        assert_eq!(vm.state.get(1, 0), Q32::ZERO);

        // Cycle 2:
        // N0 = clamp(-1 * 1 + 1) = 0.0
        // N1 = clamp(1 * 1 + 0) = 1.0
        vm.step();
        assert_eq!(vm.state.get(0, 0), Q32::ZERO);
        assert_eq!(vm.state.get(1, 0), Q32::from_f64(1.0));

        // Cycle 3:
        // N0 = clamp(-1 * 0 + 1) = 1.0
        // N1 = clamp(1 * 0 + 0) = 0.0
        vm.step();
        assert_eq!(vm.state.get(0, 0), Q32::from_f64(1.0));
        assert_eq!(vm.state.get(1, 0), Q32::ZERO);
    }

    #[test]
    fn test_vm_convergence() {
        // A latch that stabilizes in 1 cycle
        // N0 -> N0 (Identity)
        let w = Matrix::from_vec(1, 1, vec![Q32::ONE]);
        let b = Matrix::from_vec(1, 1, vec![Q32::ZERO]);
        let mut vm = Vm::new(1, Dense::new(w, b));
        vm.write_io(&[Q32::from_f64(0.75)]);

        let (cycles, converged) = vm.run_until_stable(100);
        assert!(converged);
        assert_eq!(cycles, 1);
        assert_eq!(vm.state.get(0, 0), Q32::from_f64(0.75));
    }

    #[test]
    fn test_vm_hebbian_plasticity() {
        // 2-neuron network: pre = neuron 0, dest = neuron 1
        // Initial weight W[1, 0] = 0.2
        let w = Matrix::from_vec(
            2,
            2,
            vec![Q32::ONE, Q32::ZERO, Q32::from_f64(0.2), Q32::ONE],
        );
        let b = Matrix::zeros(2, 1);
        let logic_core = Dense::new(w, b);

        let rule = PlasticityRule {
            pre: 0,
            dest: 1,
            rate: Q32::from_f64(0.1),
            decay: Q32::from_f64(0.01),
            kind: PlasticityKind::Hebbian,
        };

        let mut vm = Vm::with_plasticity(2, logic_core, vec![rule]);
        vm.write_io(&[Q32::from_f64(1.0), Q32::from_f64(1.0)]);

        // Cycle 1 plastic step
        vm.step_plastic();

        // Check adapted weight:
        // pre = 1.0, post = 1.0, cur_w = 0.2
        // hebb = 1.0, decay_term = 0.01 * 0.2 = 0.002
        // delta = 0.1 * (1.0 - 0.002) = 0.0998
        // new_w = 0.2 + 0.0998 = 0.2998 > 0.2
        let adapted_w = vm.logic_core.weights.get(1, 0);
        assert!(adapted_w > Q32::from_f64(0.2));
    }
}
