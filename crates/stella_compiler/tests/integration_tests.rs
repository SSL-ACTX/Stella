extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;
use stella_compiler::*;
use stella_core::math::Q32;
use stella_core::vm::{PlasticityKind, Vm};

#[test]
fn test_compiler_synaptic_pipeline() {
    let script = "
    in a;
    in b;
    out sum;
    out diff;

    a ~[0.5]> sum;
    b ~[0.5]> sum;

    a ~> diff;
    b ~|> diff;
    ";

    let artifact = compile(script, false, None, None).expect("Compile failed");
    let mut vm = Vm::new(artifact.state_size, artifact.core);

    vm.write_io(&[Q32::from_f64(0.8), Q32::from_f64(0.4)]);
    vm.step();

    let expected_sum =
        Q32::from_f64(0.8) * Q32::from_f64(0.5) + Q32::from_f64(0.4) * Q32::from_f64(0.5);
    assert_eq!(vm.state.get(2, 0), expected_sum);
    assert_eq!(vm.state.get(3, 0), Q32::from_f64(0.8) - Q32::from_f64(0.4));
}

#[test]
fn test_compiler_attractor_bifurcation() {
    let script = "
    in trigger;
    out alert;
    node trap : latch;

    basin Dormant {
        -> Active when trigger >= 0.75;
    }

    basin Active {
        trap = 1.0;
        trap ~> alert;
    }
    ";

    let artifact = compile(script, false, None, None).expect("Compile failed");
    let mut vm = Vm::new(artifact.state_size, artifact.core);
    vm.state.data = artifact.initial_state;

    // Inactive
    vm.state.set(0, 0, Q32::from_f64(0.2));
    vm.step();
    assert_eq!(vm.state.get(1, 0), Q32::ZERO);

    // Cross threshold to bifurcate
    vm.state.set(0, 0, Q32::from_f64(0.9));
    vm.run_until_stable(10);
    assert_eq!(vm.state.get(1, 0), Q32::ONE);
}

#[test]
fn test_compiler_arithmetics() {
    let script = "
    in a;
    out mul_out;
    out div_out;
    out neg_out;
    out not_out;

    mul_out = a * 0.5;
    div_out = a / 4.0;
    neg_out = -a;
    not_out = not a;
    ";

    let artifact = compile(script, false, None, None).expect("Compile failed");
    let mut vm = Vm::new(artifact.state_size, artifact.core);
    vm.write_io(&[Q32::from_f64(0.8)]);
    vm.step();

    assert_eq!(vm.state.get(1, 0), Q32::from_f64(0.4));
    assert_eq!(vm.state.get(2, 0), Q32::from_f64(0.2));
    assert_eq!(vm.state.get(3, 0), Q32::ZERO); // clamp01(-0.8) -> 0.0
    assert_eq!(vm.state.get(4, 0), Q32::from_f64(0.2)); // 1.0 - 0.8 -> 0.2
}

#[test]
fn test_compiler_logic_ops() {
    let script = "
    in a;
    in b;
    out xor_out;
    out nor_out;
    out neq_out;

    xor_out = a xor b;
    nor_out = a nor b;
    neq_out = a != b;
    ";

    let artifact = compile(script, false, None, None).expect("Compile failed");
    let mut vm = Vm::new(artifact.state_size, artifact.core);

    // Case 1: (1.0, 0.0) -> xor = 1, nor = 0, neq = 1
    vm.write_io(&[Q32::ONE, Q32::ZERO]);
    vm.step();
    vm.step();
    assert_eq!(vm.state.get(2, 0), Q32::ONE);
    assert_eq!(vm.state.get(3, 0), Q32::ZERO);
    assert_eq!(vm.state.get(4, 0), Q32::ONE);

    // Case 2: (1.0, 1.0) -> xor = 0, nor = 0, neq = 0
    vm.write_io(&[Q32::ONE, Q32::ONE]);
    vm.step();
    vm.step();
    assert_eq!(vm.state.get(2, 0), Q32::ZERO);
    assert_eq!(vm.state.get(3, 0), Q32::ZERO);
    assert_eq!(vm.state.get(4, 0), Q32::ZERO);

    // Case 3: (0.0, 0.0) -> xor = 0, nor = 1, neq = 0
    vm.write_io(&[Q32::ZERO, Q32::ZERO]);
    vm.step();
    vm.step();
    assert_eq!(vm.state.get(2, 0), Q32::ZERO);
    assert_eq!(vm.state.get(3, 0), Q32::ONE);
    assert_eq!(vm.state.get(4, 0), Q32::ZERO);
}

#[test]
fn test_compiler_cfg_if_else() {
    let script = "
    in cond;
    out out_val;

    if cond >= 0.5 {
        out_val = 0.8;
    } else {
        out_val = 0.2;
    }
    ";

    let artifact = compile(script, false, None, None).expect("Compile failed");
    let mut vm = Vm::new(artifact.state_size, artifact.core);

    // True branch
    vm.write_io(&[Q32::from_f64(0.9)]);
    vm.run_until_stable(10);
    assert_eq!(vm.state.get(1, 0), Q32::from_f64(0.8));

    // False branch
    vm.write_io(&[Q32::from_f64(0.1)]);
    vm.run_until_stable(10);
    assert_eq!(vm.state.get(1, 0), Q32::from_f64(0.2));
}

#[test]
fn test_compiler_cfg_while() {
    let script = "
    out done;
    node counter : latch = 0.0;

    while counter < 0.6 {
        counter += 0.2;
    }
    done = counter;
    ";

    let artifact = compile(script, false, None, None).expect("Compile failed");
    let mut vm = Vm::new(artifact.state_size, artifact.core);
    vm.state.data = artifact.initial_state;

    // Run until loop converges to stable attractor
    vm.run_until_stable(30);
    let final_val = vm.state.get(1, 0); // counter
    assert!((final_val.to_f64() - 0.6).abs() < 0.05);
}

#[test]
fn test_compiler_probe_telemetry() {
    let script = "
    in a;
    out b;
    node internal : latch = 0.5;

    b = a * 2.0;
    probe \"Output Signal\", b;
    print internal;

    if a >= 0.5 {
        probe \"High Active\", a;
    }
    ";

    let artifact = compile(script, false, None, None).expect("Compile failed");
    assert_eq!(artifact.probes.len(), 3);
    assert_eq!(artifact.probes[0].label.as_deref(), Some("Output Signal"));
    assert!(artifact.probes[0].gate_idx.is_none());
    assert_eq!(artifact.probes[1].label, None);
    assert_eq!(artifact.probes[2].label.as_deref(), Some("High Active"));
    assert!(artifact.probes[2].gate_idx.is_some());
}

#[test]
fn test_compiler_broadcasting_and_fan_in() {
    let script = "
    in s1;
    in s2;
    out d1;
    out d2;
    out sum_out;

    s1 ~> [d1, d2];
    [s1, s2] ~> sum_out;
    ";

    let artifact = compile(script, false, None, None).expect("Compile failed");
    let mut vm = Vm::new(artifact.state_size, artifact.core);

    vm.write_io(&[Q32::from_f64(0.4), Q32::from_f64(0.3)]);
    vm.step();

    // d1 (idx 2) and d2 (idx 3) should both receive s1 (0.4)
    assert_eq!(vm.state.get(2, 0), Q32::from_f64(0.4));
    assert_eq!(vm.state.get(3, 0), Q32::from_f64(0.4));
    // sum_out (idx 4) should receive s1 + s2 = 0.7
    assert_eq!(vm.state.get(4, 0), Q32::from_f64(0.7));
}

#[test]
fn test_compiler_presynaptic_gating() {
    let script = "
    in src;
    in cond;
    out dest;

    src ~> dest when cond >= 0.5;
    ";

    let artifact = compile(script, false, None, None).expect("Compile failed");
    let mut vm = Vm::new(artifact.state_size, artifact.core.clone());

    // Active gate (cond >= 0.5)
    vm.write_io(&[Q32::from_f64(0.8), Q32::from_f64(0.9)]);
    vm.run_until_stable(10);
    assert_eq!(vm.state.get(2, 0), Q32::from_f64(0.8));

    // Inactive gate (cond < 0.5)
    let mut vm2 = Vm::new(artifact.state_size, artifact.core);
    vm2.write_io(&[Q32::from_f64(0.8), Q32::from_f64(0.1)]);
    vm2.run_until_stable(10);
    assert_eq!(vm2.state.get(2, 0), Q32::ZERO);
}

#[test]
fn test_compiler_in_source_cloak() {
    let script = "
    #[cloak(pad: 32, seed: [0.25, 0.75])]

    in a;
    out b;

    b = a;
    ";

    // obfuscate argument false, but in-source cloak should trigger obfuscation with pad 32
    let artifact = compile(script, false, None, None).expect("Compile failed");
    assert_eq!(artifact.state_size, 32);
    assert!(artifact.obfuscation_key.is_some());

    let key = artifact.obfuscation_key.unwrap();
    let mut vm = Vm::new(artifact.state_size, artifact.core);

    // a = 0.75
    let clean_input = vec![Q32::from_f64(0.75), Q32::ZERO];
    let obf_input = key.encode_state(&clean_input);
    vm.write_io(&obf_input);
    vm.step();

    let raw_f64: Vec<f64> = vm.state_slice().iter().map(|q| q.to_f64()).collect();
    let decoded = key.decode_state(&raw_f64);

    // b should be 0.75
    assert!((decoded[1] - 0.75).abs() < 0.001);
}

#[test]
fn test_compiler_filter_pipeline() {
    let script = "
    in x;
    out y_step;
    out y_inv;

    y_step = x |> step;
    y_inv = x |> inv;
    ";

    let artifact = compile(script, false, None, None).expect("Compile failed");
    let mut vm = Vm::new(artifact.state_size, artifact.core);

    // x = 0.8: step should be 1.0 (clamped), inv should be 1.0 - 0.8 = 0.2
    vm.write_io(&[Q32::from_f64(0.8)]);
    vm.step();
    assert_eq!(vm.state.get(1, 0), Q32::ONE);
    assert_eq!(vm.state.get(2, 0), Q32::from_f64(0.2));

    // x = 0.2: step should be 0.0 (clamped), inv should be 1.0 - 0.2 = 0.8
    vm.write_io(&[Q32::from_f64(0.2)]);
    vm.step();
    assert_eq!(vm.state.get(1, 0), Q32::ZERO);
    assert_eq!(vm.state.get(2, 0), Q32::from_f64(0.8));
}

#[test]
fn test_compiler_bus_bundles_and_slices() {
    let script = "
    in in_bus : [2];
    out out_bus : [2];
    node internal : [2];

    // Bus to bus routing (width 2 -> width 2)
    in_bus ~> internal;

    // Individual slice routing
    out_bus[0] = internal[1];
    out_bus[1] = internal[0] * 2.0;
    ";

    let artifact = compile(script, false, None, None).expect("Compile failed");
    let mut vm = Vm::new(artifact.state_size, artifact.core);

    // in_bus[0] = 0.3, in_bus[1] = 0.7
    vm.write_io(&[Q32::from_f64(0.3), Q32::from_f64(0.7)]);
    // Step 1: in_bus propagates to internal
    vm.step();
    // Step 2: internal propagates to out_bus
    vm.step();

    // out_bus[0] should receive internal[1] (0.7)
    // out_bus[1] should receive internal[0] * 2.0 (0.3 * 2.0 = 0.6)
    // Layout: in_bus (idx 0, 1), out_bus (idx 2, 3), internal (idx 4, 5)
    assert_eq!(vm.state.get(2, 0), Q32::from_f64(0.7));
    assert_eq!(vm.state.get(3, 0), Q32::from_f64(0.6));
}

#[test]
fn test_compiler_circuit_instantiation() {
    let script = "
    circuit Filter(in sig, out result) {
        node buffer : hold;
        sig ~> buffer;
        result = buffer * 0.5;
    }

    in x;
    out y;

    f1 := Filter(x, y);
    ";

    let artifact = compile(script, false, None, None).expect("Compile failed");
    let mut vm = Vm::new(artifact.state_size, artifact.core);

    // x = 0.8
    vm.write_io(&[Q32::from_f64(0.8)]);
    // Step 1: x -> f1::buffer
    vm.step();
    // Step 2: f1::buffer * 0.5 -> y
    vm.step();

    // Layout: x (0), y (1), f1::buffer (2)
    assert_eq!(vm.state.get(1, 0), Q32::from_f64(0.4));
}

#[test]
fn test_compiler_synaptic_chaining() {
    let script = "
    in a;
    node b : hold;
    out c;

    // Chained synaptic flow: a exciting b, b exciting c
    a ~[0.5]> b ~[0.8]> c;
    ";

    let artifact = compile(script, false, None, None).expect("Compile failed");
    let mut vm = Vm::new(artifact.state_size, artifact.core);

    vm.write_io(&[Q32::from_f64(1.0)]);
    // Step 1: a (1.0) * 0.5 -> b = 0.5
    vm.step();
    assert_eq!(vm.state.get(2, 0), Q32::from_f64(0.5));

    // Step 2: b (0.5) * 0.8 -> c = 0.4
    vm.step();
    assert_eq!(vm.state.get(1, 0), Q32::from_f64(0.4));
}

#[test]
fn test_compiler_bidirectional_resonance() {
    let script = "
    node n1 : hold = 0.6;
    node n2 : hold = 0.2;

    // Mutual lateral inhibition: n1 inhibits n2, n2 inhibits n1
    n1 <~|> n2;
    ";

    let artifact = compile(script, false, None, None).expect("Compile failed");
    let mut vm = Vm::new(artifact.state_size, artifact.core);
    vm.state.data = artifact.initial_state;

    // n1 (idx 0) has weight from itself (1.0) and from n2 (-1.0)
    // n2 (idx 1) has weight from itself (1.0) and from n1 (-1.0)
    // Step 1:
    // n1_next = clamp(0.6 - 0.2)
    // n2_next = clamp(0.2 - 0.6) = 0.0
    vm.step();
    assert_eq!(vm.state.get(0, 0), Q32::from_f64(0.6) - Q32::from_f64(0.2));
    assert_eq!(vm.state.get(1, 0), Q32::ZERO);
}

#[test]
fn test_compiler_expression_presynaptic_stream() {
    let script = "
    in sig1;
    in sig2;
    out stream_out;

    // Parenthesized expression streamed directly into synaptic destination
    (sig1 * 0.5 + sig2 * 0.2 |> step) ~> stream_out;
    ";

    let artifact = compile(script, false, None, None).expect("Compile failed");
    let mut vm = Vm::new(artifact.state_size, artifact.core.clone());

    // sig1 = 0.8, sig2 = 1.0 -> 0.8 * 0.5 + 1.0 * 0.2 = 0.6 -> step >= 0.5 is 1.0
    vm.write_io(&[Q32::from_f64(0.8), Q32::from_f64(1.0)]);
    // Step 1: sig1 * 0.5 and sig2 * 0.2
    // Step 2: sum into inner activation wire
    // Step 3: step filter into stream_out
    vm.step();
    vm.step();
    vm.step();
    assert_eq!(vm.state.get(2, 0), Q32::ONE);

    // sig1 = 0.2, sig2 = 0.5 -> 0.2 * 0.5 + 0.5 * 0.2 = 0.2 -> step < 0.5 is 0.0
    let mut vm2 = Vm::new(artifact.state_size, artifact.core);
    vm2.write_io(&[Q32::from_f64(0.2), Q32::from_f64(0.5)]);
    vm2.step();
    vm2.step();
    vm2.step();
    assert_eq!(vm2.state.get(2, 0), Q32::ZERO);
}

#[test]
fn test_compiler_hebbian_plasticity() {
    let script = "
    node pre : hold = 1.0;
    node post : hold = 1.0;

    learn pre ~> post : 0.1, 0.01;
    ";

    let artifact = compile(script, false, None, None).expect("Compile failed");
    assert_eq!(artifact.plasticity_rules.len(), 1);
    let rule = &artifact.plasticity_rules[0];
    assert_eq!(rule.pre, 0);
    assert_eq!(rule.dest, 1);

    let mut vm = Vm::with_plasticity(
        artifact.state_size,
        artifact.core,
        artifact.plasticity_rules,
    );
    vm.state.data = artifact.initial_state;

    // Weight starts at 0.0
    assert_eq!(vm.logic_core.weights.get(1, 0), Q32::ZERO);

    // Step 1: pre=1.0, post=1.0 -> weight increases: 0 + 0.1 * (1.0*1.0 - 0.01*0) = 0.1
    vm.step_plastic();
    assert_eq!(vm.logic_core.weights.get(1, 0), Q32::from_f64(0.1));

    // Step 2: weight increases further
    vm.step_plastic();
    let expected_w2 = 0.1 + 0.1 * (1.0 * 1.0 - 0.01 * 0.1);
    let actual = vm.logic_core.weights.get(1, 0).to_f64();
    assert!((actual - expected_w2).abs() < 0.001);
}

#[test]
fn test_compiler_declarative_synaptic_plasticity() {
    let script = "
    node sensory : hold = 0.8;
    node associative : hold = 0.6;

    synapse sensory ~> associative {
        plastic: true;
        rule: oja;
        rate: 0.1;
        decay: 0.05;
    }
    ";

    let artifact = compile(script, false, None, None).expect("Compile failed");
    assert_eq!(artifact.plasticity_rules.len(), 1);
    let rule = &artifact.plasticity_rules[0];
    assert_eq!(rule.kind, PlasticityKind::Oja);

    let mut vm = Vm::with_plasticity(
        artifact.state_size,
        artifact.core,
        artifact.plasticity_rules,
    );
    vm.state.data = artifact.initial_state;

    // Initial weight is 0.0
    assert_eq!(vm.logic_core.weights.get(1, 0), Q32::ZERO);

    // Oja rule: dW = rate * (post * pre - post^2 * W)
    // pre = 0.8, post = 0.6, W_0 = 0.0
    // dW_1 = 0.1 * (0.6 * 0.8 - 0.36 * 0.0) = 0.1 * 0.48 = 0.048
    vm.step_plastic();
    let actual_w1 = vm.logic_core.weights.get(1, 0).to_f64();
    assert!((actual_w1 - 0.048).abs() < 0.001);
}

#[test]
fn test_compiler_stability_assertion_pass() {
    let script = "
    node a : leak(0.5) = 0.5;
    node b : leak(0.5) = 0.5;

    // Convergent negative feedback loop
    a ~[0.4]> b;
    b ~[-0.4]> a;

    #[stable]
    ";

    let res = compile(script, false, None, None);
    assert!(res.is_ok(), "Expected stable loop to compile cleanly");
}

#[test]
fn test_compiler_stability_assertion_fail() {
    let script = "
    node a : hold = 0.5;
    node b : hold = 0.5;
    a ~[1.5]> b;
    b ~[1.5]> a;
    #[stable(a)]
    ";
    let res = compile(script, false, None, None);
    assert!(res.is_err());
    assert!(res
        .unwrap_err()
        .contains("Attractor stability assertion failed"));
}

#[test]
fn test_compiler_gated_and_shunted_synapses() {
    let script = "
    in sig;
    in ctrl;
    out gated_out;
    out shunted_out;

    sig ~> (gate: ctrl) ~> gated_out;
    sig ~> [shunt: ctrl] ~> shunted_out;
    ";

    let artifact = compile(script, false, None, None).expect("Compile failed");

    // Case 1: ctrl = 1.0 (conducting gate, shunted/quenched output)
    let mut vm1 = Vm::new(artifact.state_size, artifact.core.clone());
    vm1.write_io(&[Q32::from_f64(0.8), Q32::ONE]);
    vm1.step();
    vm1.step();
    assert_eq!(vm1.state.get(2, 0), Q32::from_f64(0.8)); // gated_out is 0.8
    assert_eq!(vm1.state.get(3, 0), Q32::ZERO); // shunted_out is 0.0

    // Case 2: ctrl = 0.0 (blocked gate, un-shunted output conducts)
    let mut vm2 = Vm::new(artifact.state_size, artifact.core);
    vm2.write_io(&[Q32::from_f64(0.8), Q32::ZERO]);
    vm2.step();
    vm2.step();
    assert_eq!(vm2.state.get(2, 0), Q32::ZERO); // gated_out is 0.0
    assert_eq!(vm2.state.get(3, 0), Q32::from_f64(0.8)); // shunted_out is 0.8
}

#[test]
fn test_compiler_bifurcation() {
    let script = "
    in signal;
    out low_branch;
    out high_branch;

    bifurcate (signal) {
        < 0.5 ~> low_branch;
        >= 0.5 ~> high_branch;
    }
    ";

    let artifact = compile(script, false, None, None).expect("Compile failed");

    // When signal = 0.2 (< 0.5)
    let mut vm_low = Vm::new(artifact.state_size, artifact.core.clone());
    vm_low.write_io(&[Q32::from_f64(0.2)]);
    vm_low.step();
    vm_low.step();
    assert_eq!(vm_low.state.get(1, 0), Q32::ONE);
    assert_eq!(vm_low.state.get(2, 0), Q32::ZERO);

    // When signal = 0.8 (>= 0.5)
    let mut vm_high = Vm::new(artifact.state_size, artifact.core);
    vm_high.write_io(&[Q32::from_f64(0.8)]);
    vm_high.step();
    vm_high.step();
    assert_eq!(vm_high.state.get(1, 0), Q32::ZERO);
    assert_eq!(vm_high.state.get(2, 0), Q32::ONE);
}

#[test]
fn test_compiler_lateral_competition() {
    let script = "
    node cand_a : hold = 0.9;
    node cand_b : hold = 0.3;

    compete {
        branch_a: {
            cand_a += 0.0;
        }
        branch_b: {
            cand_b += 0.0;
        }
    } resolve winner_take_all(0.2);
    ";

    let artifact = compile(script, false, None, None).expect("Compile failed");
    let mut vm = Vm::new(artifact.state_size, artifact.core);
    vm.state.data = artifact.initial_state;

    // Cand A starts at 0.9, Cand B at 0.3
    // Due to lateral inhibition (-0.6 cross-coupling):
    // Cand A stays high, Cand B is quenched to 0.0
    vm.step();
    assert!(vm.state.get(0, 0).to_f64() > 0.5);
    assert_eq!(vm.state.get(1, 0), Q32::ZERO);
}

#[test]
fn test_compiler_oscillator_limit_cycle() {
    let script = "
    node clk : oscillator(4) = 1.0;
    ";

    let artifact = compile(script, false, None, None).expect("Compile failed");
    let mut vm = Vm::new(artifact.state_size, artifact.core);
    vm.state.data = artifact.initial_state;

    // Verify initial state: u = 1.0, quad = 0.5
    assert!((vm.state.get(0, 0).to_f64() - 1.0).abs() < 1e-4);
    assert!((vm.state.get(1, 0).to_f64() - 0.5).abs() < 1e-4);

    // Advance 4 steps (full period = 4)
    for _ in 0..4 {
        vm.step();
    }

    // Must return to initial limit-cycle orbit state (1.0, 0.5)
    assert!((vm.state.get(0, 0).to_f64() - 1.0).abs() < 1e-3);
    assert!((vm.state.get(1, 0).to_f64() - 0.5).abs() < 1e-3);
}

#[test]
fn test_compiler_full_flow_broadcast_and_fan_in() {
    let script = "
    in s;
    in g;
    in sh;
    out d1;
    out d2;
    out comb;

    s ~> (gate: g) ~> [d1, d2];
    [d1, d2] ~> [shunt: sh] ~> comb;
    ";

    let artifact = compile(script, false, None, None).expect("Compile failed");
    assert!(!artifact.symbols.is_empty());

    // Case: s = 0.8, g = 1.0, sh = 0.0
    let mut vm = Vm::new(artifact.state_size, artifact.core);
    vm.write_io(&[Q32::from_f64(0.8), Q32::ONE, Q32::ZERO]);
    vm.step();
    vm.step();
    assert_eq!(vm.state.get(3, 0), Q32::from_f64(0.8)); // d1
    assert_eq!(vm.state.get(4, 0), Q32::from_f64(0.8)); // d2
    vm.step(); // propagates from d1/d2 into shunt scratch wires
    vm.step(); // propagates from shunt scratch wires into comb
               // comb gets sum of d1 and d2 = 1.0 (clamped)
    assert_eq!(vm.state.get(5, 0), Q32::ONE); // comb
}

#[test]
fn test_compiler_tensor_multidimensional_indexing() {
    let script = "
    in image : [2, 3];
    out out_patch : [2, 3];
    node feature_map : [2, 3] : hold;

    // Route whole tensor or elementwise
    image ~> feature_map;

    // Coordinate indexing: row 0, col 2 -> flat index 0 * 3 + 2 = 2
    // Coordinate indexing: row 1, col 1 -> flat index 1 * 3 + 1 = 4
    out_patch[0, 2] = feature_map[0, 2] * 2.0;
    out_patch[1, 1] = feature_map[1, 1];
    ";

    let artifact = compile(script, false, None, None).expect("Compile failed");
    let mut vm = Vm::new(artifact.state_size, artifact.core);

    // image has 6 neurons: (0,0), (0,1), (0,2), (1,0), (1,1), (1,2)
    // flat index 2 is (0,2), flat index 4 is (1,1)
    let mut input = vec![Q32::ZERO; 6];
    input[2] = Q32::from_f64(0.25); // image[0, 2]
    input[4] = Q32::from_f64(0.8); // image[1, 1]
    vm.write_io(&input);

    // Step 1: image propagates to feature_map
    vm.step();
    // Step 2: feature_map propagates to out_patch
    vm.step();

    // out_patch starts at index 6 (image is 0..6, out_patch is 6..12, feature_map is 12..18)
    // out_patch[0, 2] is 6 + 2 = 8: should be 0.25 * 2.0 = 0.5
    // out_patch[1, 1] is 6 + 4 = 10: should be 0.8
    assert_eq!(vm.state.get(8, 0), Q32::from_f64(0.5));
    assert_eq!(vm.state.get(10, 0), Q32::from_f64(0.8));
}

#[test]
fn test_compiler_bus_range_slicing() {
    let script = "
    in full_in : [8];
    out upper_out : [4];
    out lower_out : [4];

    // Sub-slice wiring: top 4 wires and bottom 4 wires
    full_in[4..8] ~> upper_out[0..4];
    full_in[0..4] ~[0.5]> lower_out[0..4];
    ";

    let artifact = compile(script, false, None, None).expect("Compile failed");
    let mut vm = Vm::new(artifact.state_size, artifact.core);

    let mut input = vec![Q32::ZERO; 8];
    input[0] = Q32::from_f64(0.6); // lower
    input[6] = Q32::from_f64(0.9); // upper: index 6 = 4 + 2 -> upper_out[2]
    vm.write_io(&input);

    vm.step();

    // upper_out starts at index 8 (full_in 0..8, upper_out 8..12, lower_out 12..16)
    // upper_out[2] = full_in[6] = 0.9 -> index 8 + 2 = 10
    // lower_out[0] = full_in[0] * 0.5 = 0.6 * 0.5 = 0.3 -> index 12 + 0 = 12
    assert_eq!(vm.state.get(10, 0), Q32::from_f64(0.9));
    assert_eq!(vm.state.get(12, 0), Q32::from_f64(0.3));
}

#[test]
fn test_compiler_stability_hold_nodes_are_excluded() {
    let script = "
    in sensor;
    out readout;
    node buf : hold;
    sensor ~> buf;
    buf ~> readout;
    #[stable]
    ";
    let res = compile(script, false, None, None);
    assert!(
        res.is_ok(),
        "hold pipeline must pass assert stable; got: {:?}",
        res.err()
    );
}

#[test]
fn test_compiler_stability_hold_pipeline_passes() {
    let script = "
    in inp : [4];
    out filtered : [4];
    node buf : [4] : hold;
    inp ~> buf;
    buf ~> filtered;
    #[stable]
    ";
    let res = compile(script, false, None, None);
    assert!(
        res.is_ok(),
        "4-wide hold bus pipeline must pass assert stable; got: {:?}",
        res.err()
    );
}

#[test]
fn test_compiler_stability_targeted_hold_runaway_fails() {
    let script = "
    node dangerous_a : hold = 0.5;
    node dangerous_b : hold = 0.5;
    dangerous_a ~[1.5]> dangerous_b;
    dangerous_b ~[1.5]> dangerous_a;
    #[stable(dangerous_a)]
    ";
    let res = compile(script, false, None, None);
    assert!(
        res.is_err(),
        "targeted runaway hold loop must fail assert stable"
    );
    assert!(res
        .unwrap_err()
        .contains("Attractor stability assertion failed"));
}

#[test]
fn test_compiler_stability_mixed_hold_and_leak_passes() {
    let script = "
    in sig;
    out out_sig;
    node buf : hold;
    node integrator : leak(0.5) = 0.5;
    sig ~> buf;
    buf ~[0.4]> integrator;
    integrator ~[-0.3]> integrator;
    integrator ~> out_sig;
    #[stable]
    ";
    let res = compile(script, false, None, None);
    assert!(
        res.is_ok(),
        "mixed hold + convergent leak must pass assert stable; got: {:?}",
        res.err()
    );
}

#[test]
fn test_compiler_tensor_range_slice_and_diagonal_readout() {
    let script = "
    in retina : [4, 4];
    out patch_readout : [4];
    out band_readout : [2];
    node feature_grid : [4, 4] : leak(0.5);
    retina ~> feature_grid;
    retina[0..4] ~> patch_readout[0..4];
    band_readout[0] = feature_grid[0, 0] * 0.5 + feature_grid[1, 1] * 0.3 + feature_grid[2, 2] * 0.2;
    band_readout[1] = (feature_grid[0, 1] + feature_grid[1, 2] + feature_grid[2, 3]) * 0.333;
    #[stable]
    ";

    let artifact = compile(script, false, None, None).expect("Compile failed");

    let find = |name: &str| {
        artifact
            .symbols
            .iter()
            .find(|s| s.name == name)
            .unwrap()
            .index
    };
    let retina_base = find("retina");
    let patch_readout_base = find("patch_readout");
    let band_readout_base = find("band_readout");

    let mut vm = Vm::new(artifact.state_size, artifact.core);
    vm.state.data = artifact.initial_state;

    let mut input = vec![Q32::ZERO; 16];
    input[0] = Q32::ONE;
    input[1] = Q32::from_f64(0.8);
    vm.write_io(&input);

    vm.step();
    vm.step();
    vm.step();
    vm.step();
    // band_readout[0] = feature_grid[0,0]*0.5 = 1.0*0.5 = 0.5 after convergence
    let band0 = vm.state.get(band_readout_base, 0).to_f64();
    assert!(
        (band0 - 0.5).abs() < 0.01,
        "band_readout[0] expected ~0.5, got {band0}"
    );

    assert_eq!(
        vm.state.get(patch_readout_base, 0),
        Q32::ONE,
        "patch_readout[0] must be 1.0"
    );
    assert_eq!(
        vm.state.get(patch_readout_base + 1, 0),
        Q32::from_f64(0.8),
        "patch_readout[1] must be 0.8"
    );
    let _ = retina_base; // used implicitly via write_io
}

#[test]
fn test_compiler_bus_range_slice_isolation() {
    let script = "
    in full_in : [8];
    out upper_out : [4];
    out lower_out : [4];
    full_in[4..8] ~> upper_out[0..4];
    full_in[0..4] ~> lower_out[0..4];
    ";

    let artifact = compile(script, false, None, None).expect("Compile failed");

    let find = |name: &str| {
        artifact
            .symbols
            .iter()
            .find(|s| s.name == name)
            .unwrap()
            .index
    };
    let upper_base = find("upper_out");
    let lower_base = find("lower_out");

    let mut vm = Vm::new(artifact.state_size, artifact.core);

    let mut input = vec![Q32::ZERO; 8];
    input[4] = Q32::from_f64(0.7);
    input[7] = Q32::from_f64(0.3);
    vm.write_io(&input);
    vm.step();

    assert_eq!(
        vm.state.get(upper_base, 0),
        Q32::from_f64(0.7),
        "upper_out[0] must be 0.7"
    );
    assert_eq!(
        vm.state.get(upper_base + 3, 0),
        Q32::from_f64(0.3),
        "upper_out[3] must be 0.3"
    );
    for i in 0..4 {
        assert_eq!(
            vm.state.get(lower_base + i, 0),
            Q32::ZERO,
            "lower_out[{i}] must be zero"
        );
    }
}

#[test]
fn test_compiler_countermeasure_1_interface_tomography_defense() {
    // Circuit modeling the attractor control flow:
    let script = "
    #[cloak(pad: 32, seed: [0.1337, 0.7331])]

    in sensor_temp;
    in override_kill;
    out critical_alarm;

    // alarm activates if sensor_temp is high and override is off
    critical_alarm = sensor_temp * 1.5;
    ";

    let artifact = compile(script, false, None, None).expect("Compile failed");
    assert!(artifact.obfuscation_key.is_some());
    let key = artifact.obfuscation_key.unwrap();
    assert!(
        key.rolling.is_some(),
        "Dynamic terminal rolling must be active"
    );

    // Fuzzer provides identical inputs across distinct continuous clock cycles:
    let clean_output = vec![0.85]; // simulated settled alarm output

    let mut observed_samples = Vec::new();
    for t in 0..20 {
        let obs = key.observe_terminals(&clean_output, t);
        observed_samples.push(obs[0]);

        // Exact bit-perfect recovery for authorized decoder:
        let decoded = key.decode_terminals(&obs, t);
        assert!(
            (decoded[0] - clean_output[0]).abs() < 1e-6,
            "Decoder with key at t={} must bit-perfectly recover clean output",
            t
        );
    }

    // Verify non-stationarity / zero algebraic linearity for black-box fuzzers:
    // All observed samples across different t are non-identical:
    let unique_samples: alloc::collections::BTreeSet<u64> =
        observed_samples.iter().map(|&f| f.to_bits()).collect();
    assert_eq!(
        unique_samples.len(),
        20,
        "Black-box fuzzer must observe high-entropy non-stationary values across cycles"
    );
}

#[test]
fn test_compiler_circuit_return_and_expression_call() {
    let script = "
    circuit Double(x) -> y {
        node internal_mult;
        internal_mult = x * 0.5;
        y = internal_mult;
    }

    in a;
    out b;

    b = Double(a) + 0.2;
    ";
    let artifact = compile(script, false, None, None).expect("Circuit call should compile");
    let mut vm = Vm::new(artifact.state_size, artifact.core);
    let input = vec![Q32::from_f64(0.8)];
    vm.write_io(&input);
    // Step 1: a -> _call_Double_0::internal_mult (0.8 * 0.5 = 0.4)
    // Step 2: _call_Double_0::internal_mult -> _call_Double_0::y (0.4)
    // Step 3: _call_Double_0::y + 0.2 -> b (0.4 + 0.2 = 0.6)
    vm.step();
    vm.step();
    vm.step();

    // Double(0.8) + 0.2 = 0.4 + 0.2 = 0.6
    let b_val = vm.state.get(1, 0); // b is index 1
    assert!((b_val.0 - Q32::from_f64(0.6).0).abs() <= 1);
}

#[test]
fn test_compiler_circuit_return_nested_and_template() {
    let script = "
    circuit Scale<factor = 0.5>(x) -> out_val {
        out_val = x * factor;
    }

    in x;
    out res;

    res = Scale<0.5>(Scale<0.5>(x));
    ";
    let artifact =
        compile(script, false, None, None).expect("Nested template circuit calls should compile");
    let mut vm = Vm::new(artifact.state_size, artifact.core);
    let input = vec![Q32::from_f64(0.8)];
    vm.write_io(&input);
    // Step 1: inner Scale: x -> _call_Scale_0::out_val (0.8 * 0.5 = 0.4)
    // Step 2: outer Scale: _call_Scale_0::out_val -> _call_Scale_1::out_val (0.4 * 0.5 = 0.2)
    // Step 3: res = _call_Scale_1::out_val (0.2)
    vm.step();
    vm.step();
    vm.step();

    // Scale<0.5>(Scale<0.5>(0.8)) = 0.2
    let res_val = vm.state.get(1, 0);
    assert_eq!(res_val, Q32::from_f64(0.2));
}

#[test]
fn test_compiler_universal_shapes_and_type_aliases() {
    let script = "
    type Nibble = [4];

    in a : Nibble;
    in b : [4];
    out sum : Nibble;

    sum[0] = a[0] + b[0];
    sum[1] = a[1] + b[1];
    sum[2] = a[2] + b[2];
    sum[3] = a[3] + b[3];
    ";
    let artifact =
        compile(script, false, None, None).expect("Type alias and universal shape should compile");
    let mut vm = Vm::new(artifact.state_size, artifact.core);

    // a = 4 neurons, b = 4 neurons, sum = 4 neurons
    let input = vec![
        Q32::from_f64(0.1),
        Q32::from_f64(0.2),
        Q32::from_f64(0.3),
        Q32::from_f64(0.4),
        Q32::from_f64(0.05),
        Q32::from_f64(0.10),
        Q32::from_f64(0.15),
        Q32::from_f64(0.20),
    ];
    vm.write_io(&input);
    vm.step();

    // Terminal out 'sum' starts at index 8
    for i in 0..4 {
        let expected = input[i] + input[4 + i];
        assert_eq!(vm.state.get(8 + i, 0), expected);
    }
}

#[test]
fn test_compiler_neural_addressing_ram() {
    let script = "
    in addr : [2];
    in val_in;
    out data_out;

    node ram : [4] : hold;

    // Initialize RAM memory cells
    ram[0] = 0.25;
    ram[1] = 0.50;
    ram[2] = 0.75;
    ram[3] = 1.00;

    // Continuous 1-hot dynamic read
    data_out = ram[addr];
    ";

    let artifact =
        compile(script, false, None, None).expect("Neural addressing RAM script should compile");
    let mut vm = Vm::new(artifact.state_size, artifact.core);

    // Test reading addr = 2 (binary 10: bit0 = 0.0, bit1 = 1.0)
    // addr is 2 neurons, val_in is 1 neuron (indices 0, 1, 2)
    // data_out is at index 3
    vm.write_io(&[Q32::ZERO, Q32::ONE, Q32::ZERO]);
    for _ in 0..5 {
        vm.step();
    }

    let read_val = vm.state.get(3, 0);
    let expected = Q32::from_f64(0.75);
    assert!(
        (read_val.0 - expected.0).abs() <= 2000,
        "Expected ~0.75, got {:?}",
        read_val.to_f64()
    );

    // Test reading addr = 1 (binary 01: bit0 = 1.0, bit1 = 0.0)
    vm.write_io(&[Q32::ONE, Q32::ZERO, Q32::ZERO]);
    for _ in 0..5 {
        vm.step();
    }
    let read_val2 = vm.state.get(3, 0);
    let expected2 = Q32::from_f64(0.50);
    assert!(
        (read_val2.0 - expected2.0).abs() <= 2000,
        "Expected ~0.50, got {:?}",
        read_val2.to_f64()
    );
}

#[test]
fn test_compiler_piecewise_curve_and_pattern_match() {
    let script = "
    in sensor;
    out curve_out;
    in packet : [6];
    out match_out;

    // Piecewise transfer curve
    curve_out = sensor |> curve([
        0.0 => 0.0,
        0.5 => 1.0,
        1.0 => 0.0,
    ]);

    // Orthogonal pattern match against ascii 'STELLA'
    match_out = packet |> match(\"STELLA\");
    ";

    let artifact = compile(script, false, None, None)
        .expect("Piecewise curve and pattern matching script should compile");
    let mut vm = Vm::new(artifact.state_size, artifact.core);

    // Test curve at sensor = 0.5 -> curve_out should be ~1.0
    // sensor is index 0, curve_out is terminal out at index 7
    // packet is indices 1..7
    let mut input = vec![Q32::ZERO; 7];
    input[0] = Q32::from_f64(0.5);

    // packet exact match for 'STELLA' (S=83, T=84, E=69, L=76, L=76, A=65)
    let stella_bytes = b"STELLA";
    for (i, &b) in stella_bytes.iter().enumerate() {
        input[1 + i] = Q32::from_f64((b as f64) / 255.0);
    }

    vm.write_io(&input);
    for _ in 0..5 {
        vm.step();
    }

    // Check curve_out (first output terminal, index 7)
    let curve_val = vm.state.get(7, 0).to_f64();
    assert!(
        (curve_val - 1.0).abs() <= 0.05,
        "Expected curve_out ~1.0, got {:.4}",
        curve_val
    );

    // Check match_out (second output terminal, index 8)
    let match_val = vm.state.get(8, 0).to_f64();
    assert!(
        (match_val - 1.0).abs() <= 0.05,
        "Expected match_out ~1.0, got {:.4}",
        match_val
    );
}

#[test]
fn test_compiler_phase_automata_and_attractor_basins() {
    let script = "
    in trigger;
    out alert;
    node trap : latch;

    phase Controller {
        state Idle {
            -> Active when trigger >= 0.75;
        }

        state Active {
            trap = 1.0;
            trap ~> alert;
        }
    }
    ";

    let artifact =
        compile(script, false, None, None).expect("Phase automata script should compile");
    let mut vm = Vm::new(artifact.state_size, artifact.core);
    vm.state.data = artifact.initial_state;

    // Initially in Idle: trigger low (0.2)
    vm.state.set(0, 0, Q32::from_f64(0.2));
    vm.step();
    assert_eq!(vm.state.get(1, 0), Q32::ZERO);

    // Trigger high (0.9) -> bifurcates from Idle to Active
    vm.state.set(0, 0, Q32::from_f64(0.9));
    vm.run_until_stable(10);
    assert_eq!(vm.state.get(1, 0), Q32::ONE);
}

#[test]
fn test_compiler_circuit_interface_shape_contracts() {
    // Valid shape contract match:
    let valid_script = "
    circuit ConvBlock(in img [2, 2], out feat [2, 2]) {
        feat[0, 0] = img[0, 0] * 0.5;
        feat[0, 1] = img[0, 1] * 0.5;
        feat[1, 0] = img[1, 0] * 0.5;
        feat[1, 1] = img[1, 1] * 0.5;
    }

    in input_matrix : [2, 2];
    out output_matrix : [2, 2];

    blk := ConvBlock(input_matrix, output_matrix);
    ";

    let artifact = compile(valid_script, false, None, None)
        .expect("Valid tensor shape contract should compile cleanly");
    let mut vm = Vm::new(artifact.state_size, artifact.core);

    let input = vec![
        Q32::from_f64(0.4),
        Q32::from_f64(0.6),
        Q32::from_f64(0.8),
        Q32::ONE,
    ];
    vm.write_io(&input);
    vm.step();

    // Output starts at index 4 (input_matrix is 0..4, output_matrix is 4..8)
    assert_eq!(vm.state.get(4, 0), Q32::from_f64(0.2));
    assert_eq!(vm.state.get(5, 0), Q32::from_f64(0.3));
    assert_eq!(vm.state.get(6, 0), Q32::from_f64(0.4));
    assert_eq!(vm.state.get(7, 0), Q32::from_f64(0.5));

    // Mismatched shape contract: expected tensor[2, 2], provided [2, 3]
    let invalid_script = "
    circuit ConvBlock(in img [2, 2], out feat [2, 2]) {
        feat[0, 0] = img[0, 0];
    }

    in wrong_matrix : [2, 3];
    out output_matrix : [2, 2];

    blk := ConvBlock(wrong_matrix, output_matrix);
    ";

    let res = compile(invalid_script, false, None, None);
    assert!(
        res.is_err(),
        "Mismatched tensor shape contract must fail compilation"
    );
    let err = res.unwrap_err();
    assert!(
        err.contains("shape mismatch") || err.contains("width mismatch"),
        "Error should report shape/width mismatch: {}",
        err
    );
}

#[test]
fn test_compiler_multibranch_flow_fanout() {
    let script = "
    in sensor;
    out fast_path;
    out inverted_path;
    out scaled_path;

    sensor ~> (
        |> saturate ~> fast_path,
        |> inv ~> inverted_path,
        * 0.5 ~> scaled_path
    );
    ";

    let artifact =
        compile(script, false, None, None).expect("Multi-branch pipeline should compile");
    let mut vm = Vm::new(artifact.state_size, artifact.core);

    // Test with sensor = 0.8:
    // fast_path = saturate(0.8) = 1.0 (clamped saturating activation)
    // inverted_path = inv(0.8) = 0.2
    // scaled_path = 0.8 * 0.5 = 0.4
    vm.write_io(&[Q32::from_f64(0.8)]);
    vm.step();
    vm.step();

    // sensor is 0, fast_path is 1, inverted_path is 2, scaled_path is 3
    assert_eq!(vm.state.get(1, 0), Q32::ONE);
    assert_eq!(vm.state.get(2, 0), Q32::from_f64(0.2));
    assert_eq!(vm.state.get(3, 0), Q32::from_f64(0.4));
}

#[test]
fn test_compiler_multibranch_chained_pipeline_and_nested_fanout() {
    let script = "
    in sensor;
    out out1;
    out out2;
    out out3;

    sensor ~> (
        |> inv * 0.5 ~> (out1, out2),
        * 2.0 |> clamp ~> out3
    );
    ";

    let artifact = compile(script, false, None, None).expect("Chained multi-branch should compile");
    let mut vm = Vm::new(artifact.state_size, artifact.core);

    // sensor = 0.2:
    // branch 1: inv(0.2) = 0.8; 0.8 * 0.5 = 0.4. Fanout to out1 and out2.
    // branch 2: 0.2 * 2.0 = 0.4; clamp(0.4) = 0.4. Sent to out3.
    vm.write_io(&[Q32::from_f64(0.2)]);
    for _ in 0..5 {
        vm.step();
    }

    assert_eq!(vm.state.get(1, 0), Q32::from_f64(0.4));
    assert_eq!(vm.state.get(2, 0), Q32::from_f64(0.4));
    assert_eq!(vm.state.get(3, 0), Q32::from_f64(0.4));
}

#[test]
fn test_compiler_spatial_convolution_conv2d() {
    let script = "
    const SOBEL_X = [
        [-1.0, 0.0, 1.0],
        [-2.0, 0.0, 2.0],
        [-1.0, 0.0, 1.0]
    ];

    in retina : [4, 4];
    out edges : [2, 2];

    edges = conv2d(retina, SOBEL_X, stride: 1);
    ";

    let artifact = compile(script, false, None, None).expect("Conv2d program should compile");
    let mut vm = Vm::new(artifact.state_size, artifact.core);

    // Provide a 4x4 input patch: vertical edge step
    // [0, 0, 1, 1]
    // [0, 0, 1, 1]
    // [0, 0, 1, 1]
    // [0, 0, 1, 1]
    let mut patch = vec![Q32::ZERO; 16];
    for r in 0..4 {
        for c in 0..4 {
            if c >= 2 {
                patch[r * 4 + c] = Q32::ONE;
            }
        }
    }

    vm.write_io(&patch);
    vm.step();

    // edges is [2, 2] located starting at index 16.
    // For receptive field at out [0, 0] (retina rows 0..3, cols 0..3):
    // col 0: 0, col 1: 0, col 2: 1
    // Filter applied to col 0 is -1, -2, -1 (sum 0)
    // Filter applied to col 1 is 0
    // Filter applied to col 2 is +1*1 + 2*1 + 1*1 = 4.0, clamped to 1.0
    let out_00 = vm.state.get(16, 0).to_f64();
    assert_eq!(out_00, 1.0, "Expected saturated edge response at [0, 0]");
}

#[test]
fn test_compiler_conv2d_same_padding_and_avgpool2d() {
    let script = "
    const BOX = [
        [1.0, 1.0],
        [1.0, 1.0]
    ];

    in raw : [4, 4];
    node blurred : [4, 4];
    out downsampled : [2, 2];

    blurred = conv2d(raw, BOX, stride: 1, padding: same);
    downsampled = avgpool2d(blurred, kernel_size: 2, stride: 2);
    ";

    let artifact = compile(script, false, None, None)
        .expect("Conv2d with same padding and AvgPool2d should compile");
    let mut vm = Vm::new(artifact.state_size, artifact.core);

    // Constant 1.0 input over 4x4
    let patch = vec![Q32::from_f64(0.25); 16];
    vm.write_io(&patch);
    vm.step(); // conv2d computes blurred
    vm.step(); // avgpool2d computes downsampled

    // downsampled is at out terminals index 16..20
    let out_val = vm.state.get(16, 0).to_f64();
    assert!(out_val > 0.0, "Downsampled output should be positive");
}

#[test]
fn test_compiler_continuous_basin_energy_readout() {
    let script = "
    in error;
    out gain;

    phase Controller {
        state Searching {
            -> Locked when error < 0.2;
        }
        state Locked {
            -> Searching when error > 0.8;
        }
    }

    gain = state.Searching * 0.8 + state.Locked * 0.1;
    ";

    let artifact = compile(script, false, None, None).expect("Basin energy readout should compile");
    let mut vm = Vm::new(artifact.state_size, artifact.core);
    vm.state.data = artifact.initial_state;

    // Initial state: Searching basin energy = 1.0, Locked = 0.0
    // gain = 1.0 * 0.8 + 0.0 * 0.1 = 0.8
    vm.write_io(&[Q32::from_f64(0.5)]);
    vm.step();
    vm.step();

    // gain is terminal out at index 1
    let gain_val = vm.state.get(1, 0).to_f64();
    assert!(
        (gain_val - 0.8).abs() < 0.05,
        "Expected initial gain around 0.8, got {}",
        gain_val
    );

    // Provide error < 0.2 (e.g. 0.1) -> bifurcation triggers transition to Locked state
    vm.write_io(&[Q32::from_f64(0.1)]);
    vm.run_until_stable(10);

    // Now Locked basin energy = 1.0, Searching = 0.0
    // gain = 0.0 * 0.8 + 1.0 * 0.1 = 0.1
    let gain_locked = vm.state.get(1, 0).to_f64();
    assert!(
        (gain_locked - 0.1).abs() < 0.05,
        "Expected locked gain around 0.1, got {}",
        gain_locked
    );
}

#[test]
fn test_compiler_plastic_and_competitive_arrows() {
    let source = r#"
        in pre;
        out post;
        node winner_a;
        node winner_b;

        // Plastic Hebbian and anti-Hebbian arrow syntax
        pre ~+(0.08, decay=0.002)+> post;

        // Mutual lateral competitive coupling shorthand
        winner_a <|(-0.75)|> winner_b;
    "#;

    let artifact = compile(source, false, None, None).expect("Compilation must succeed");
    assert_eq!(artifact.plasticity_rules.len(), 1);
    assert_eq!(artifact.plasticity_rules[0].kind, PlasticityKind::Hebbian);
    assert!((artifact.plasticity_rules[0].rate.to_f64() - 0.08).abs() < 1e-4);
    assert!((artifact.plasticity_rules[0].decay.to_f64() - 0.002).abs() < 1e-4);

    // Find winner_a and winner_b in symbols
    let wa_idx = artifact
        .symbols
        .iter()
        .find(|s| s.name == "winner_a")
        .unwrap()
        .index;
    let wb_idx = artifact
        .symbols
        .iter()
        .find(|s| s.name == "winner_b")
        .unwrap()
        .index;

    // Reciprocal weights should be -0.75
    assert!((artifact.core.weights.get(wb_idx, wa_idx).to_f64() - (-0.75)).abs() < 1e-3);
    assert!((artifact.core.weights.get(wa_idx, wb_idx).to_f64() - (-0.75)).abs() < 1e-3);
}

#[test]
fn test_compiler_energy_basin_minimization() {
    let source = r#"
        node x;
        node y;

        basin attractor_well {
            // Quadratic energy well minimization: E = (x - y)^2
            minimize (x - y) * (x - y);
        }
    "#;

    let artifact =
        compile(source, false, None, None).expect("Energy well compilation must succeed");

    let x_idx = artifact
        .symbols
        .iter()
        .find(|s| s.name == "x")
        .unwrap()
        .index;
    let y_idx = artifact
        .symbols
        .iter()
        .find(|s| s.name == "y")
        .unwrap()
        .index;

    // Both x and y should have positive mutual coupling from the quadratic energy gradient
    assert!(artifact.core.weights.get(x_idx, y_idx).to_f64() > 0.0);
    assert!(artifact.core.weights.get(y_idx, x_idx).to_f64() > 0.0);
}

#[test]
fn test_compiler_tensor_matmul_contraction() {
    let source = r#"
        const W = [
            [0.2, 0.4, 0.1],
            [0.5, 0.3, 0.8]
        ];

        in x : [2];
        out y : [3];

        y = x @* W;
    "#;

    let artifact = compile(source, false, None, None).expect("Matmul contraction must succeed");
    let mut vm = Vm::new(artifact.state_size, artifact.core);

    // Provide input x = [0.5, 0.5]
    // Expected y = [0.5, 0.5] * W
    // y[0] = 0.5*0.2 + 0.5*0.5 = 0.35
    // y[1] = 0.5*0.4 + 0.5*0.3 = 0.35
    // y[2] = 0.5*0.1 + 0.5*0.8 = 0.45
    vm.write_io(&[Q32::from_f64(0.5), Q32::from_f64(0.5)]);
    vm.step();

    let y_0 = vm.state.get(2, 0).to_f64();
    let y_1 = vm.state.get(3, 0).to_f64();
    let y_2 = vm.state.get(4, 0).to_f64();

    assert!(
        (y_0 - 0.35).abs() < 0.05,
        "Expected y[0] ~ 0.35, got {}",
        y_0
    );
    assert!(
        (y_1 - 0.35).abs() < 0.05,
        "Expected y[1] ~ 0.35, got {}",
        y_1
    );
    assert!(
        (y_2 - 0.45).abs() < 0.05,
        "Expected y[2] ~ 0.45, got {}",
        y_2
    );
}

#[test]
fn test_compiler_fully_symbolic_program() {
    let source = r#"
        const K = [
            [0.5, 0.5],
            [0.2, 0.8]
        ];

        // Pure symbolic entry ports (+>), exit ports (=>), and state nodes (*)
        +> retina[2];
        => motor[2];
        *integrator[2] ~ 0.95;
        *clock[2] @ 40;

        // Symbolic tensor contraction & flow
        integrator = retina @* K;
        integrator ~> motor;

        // Symbolic telemetry (? for probe, ! for drain)
        ? "motor_readout" : motor;
        !integrator;

        // Symbolic basin (§ for basin, ∇ for minimize)
        § attractor_ground {
            ∇ (retina[0] - retina[1]) * (retina[0] - retina[1]);
        }
    "#;

    let artifact = compile(source, false, None, None)
        .expect("Fully symbolic Stella program should compile cleanly");
    assert_eq!(artifact.probes.len(), 1);
    assert_eq!(artifact.probes[0].label.as_deref(), Some("motor_readout"));

    let ret_info = artifact
        .symbols
        .iter()
        .find(|s| s.name == "retina")
        .unwrap();
    let mot_info = artifact.symbols.iter().find(|s| s.name == "motor").unwrap();
    let int_info = artifact
        .symbols
        .iter()
        .find(|s| s.name == "integrator")
        .unwrap();
    let clk_info = artifact.symbols.iter().find(|s| s.name == "clock").unwrap();

    assert_eq!(ret_info.width, 2);
    assert_eq!(mot_info.width, 2);
    assert_eq!(int_info.width, 2);
    assert_eq!(clk_info.width, 2);

    // Verify clock is synthesized as 2D harmonic rotation oscillator
    let c0 = clk_info.index;
    let c1 = clk_info.index + 1;
    assert!(artifact.core.weights.get(c0, c0).to_f64() > 0.0);
    assert!(artifact.core.weights.get(c0, c1).to_f64() < 0.0);
}

#[test]
fn test_compiler_unified_pipeline_arrow_composition() {
    let script = r#"
    in sig;
    out out1;
    out out2;

    sig |> relu ~[0.5]> out1;
    sig |> clamp * 2.0 ~> out2;
    "#;

    let artifact =
        compile(script, false, None, None).expect("Pipeline arrow composition compile failed");
    let mut vm = Vm::new(artifact.state_size, artifact.core);

    // Provide 0.8 to input
    vm.write_io(&[Q32::from_f64(0.8)]);
    vm.run_until_stable(5);

    // out1 should be relu(0.8) * 0.5 = 0.4
    assert_eq!(vm.state.get(1, 0), Q32::from_f64(0.4));
    // out2: clamp(0.8 * 2.0) -> saturated at Q32::ONE (1.0) because neurons clamp to [0.0, 1.0]
    assert_eq!(vm.state.get(2, 0), Q32::ONE);
}

#[test]
fn test_compiler_neuromodulated_3factor_plasticity() {
    let script = r#"
    node pre : hold = 1.0;
    node post : hold = 1.0;
    node dopamine : hold = 0.0;

    pre ~+(0.1, 0.01)+> post @ dopamine;
    "#;

    let artifact =
        compile(script, false, None, None).expect("Neuromodulated plasticity compile failed");
    assert_eq!(artifact.plasticity_rules.len(), 1);
    let rule = &artifact.plasticity_rules[0];
    assert!(rule.modulator.is_some());

    let mut vm = Vm::with_plasticity(
        artifact.state_size,
        artifact.core,
        artifact.plasticity_rules,
    );
    vm.state.data = artifact.initial_state;

    // Initially dopamine = 0.0 -> weights should NOT change
    vm.step_plastic();
    assert_eq!(vm.logic_core.weights.get(1, 0), Q32::ZERO);

    // Now release dopamine = 1.0 -> weights adapt via Hebbian correlation
    let dop_idx = artifact
        .symbols
        .iter()
        .find(|s| s.name == "dopamine")
        .unwrap()
        .index;
    vm.state.set(dop_idx, 0, Q32::ONE);
    vm.step_plastic();
    assert_eq!(vm.logic_core.weights.get(1, 0), Q32::from_f64(0.1));
}

#[test]
fn test_compiler_self_basin_transition() {
    let script = r#"
    in sensor;
    out alert;
    node flag : latch;

    basin Searching {
        self ~> Locked when sensor >= 0.8;
    }

    basin Locked {
        flag = 1.0;
        flag ~> alert;
    }
    "#;

    let artifact =
        compile(script, false, None, None).expect("Self basin transition compile failed");
    let mut vm = Vm::new(artifact.state_size, artifact.core);
    vm.state.data = artifact.initial_state;

    // sensor low (0.3) -> stays in Searching
    vm.state.set(0, 0, Q32::from_f64(0.3));
    vm.step();
    assert_eq!(vm.state.get(1, 0), Q32::ZERO);

    // sensor high (0.9) -> transitions to Locked
    vm.state.set(0, 0, Q32::from_f64(0.9));
    vm.run_until_stable(10);
    assert_eq!(vm.state.get(1, 0), Q32::ONE);
}

#[test]
fn test_compiler_bidirectional_plasticity() {
    let script = r#"
    +> node_a;
    +> node_b;
    node_a <~+(0.05, 0.01)+~> node_b;
    "#;

    let artifact = compile(script, false, None, None)
        .expect("Bidirectional plasticity connection should compile cleanly");
    assert_eq!(artifact.plasticity_rules.len(), 2);
    let r1 = &artifact.plasticity_rules[0];
    let r2 = &artifact.plasticity_rules[1];
    assert_eq!(r1.pre, 0);
    assert_eq!(r1.dest, 1);
    assert_eq!(r2.pre, 1);
    assert_eq!(r2.dest, 0);
}

#[test]
fn test_compiler_tensor_outer_product() {
    let script = r#"
    +> pre[2];
    +> post[2];
    node W[2, 2];
    W = pre ^* post;
    "#;

    let artifact = compile(script, false, None, None)
        .expect("Outer product contraction should compile cleanly");
    let mut vm = Vm::new(artifact.state_size, artifact.core);
    vm.state.data = artifact.initial_state;

    let pre_info = artifact.symbols.iter().find(|s| s.name == "pre").unwrap();
    let post_info = artifact.symbols.iter().find(|s| s.name == "post").unwrap();
    let w_info = artifact.symbols.iter().find(|s| s.name == "W").unwrap();

    vm.state.set(pre_info.index, 0, Q32::ONE);
    vm.state.set(pre_info.index + 1, 0, Q32::ONE);
    vm.state.set(post_info.index, 0, Q32::ONE);
    vm.state.set(post_info.index + 1, 0, Q32::ZERO);

    // Two steps: step 1 computes scratch mult_wire, step 2 propagates to W
    vm.step();
    vm.step();

    let w_base = w_info.index;
    assert_eq!(vm.state.get(w_base, 0), Q32::ONE);
    assert_eq!(vm.state.get(w_base + 1, 0), Q32::ZERO);
}

#[test]
fn test_compiler_module_imports_aliased_and_selective() {
    use bumpalo::Bump;
    use stella_frontend::importer::MemoryResolver;
    use stella_frontend::parse_synaptic_with_resolver;

    let mut resolver = MemoryResolver::new();
    resolver.add_file(
        "math/filters.stl",
        r#"
        circuit Lowpass(+> sig, => filtered) {
            *smoothed ~ 0.9
            smoothed = sig * 0.5
            smoothed ~> filtered
        }
        "#,
    );

    let main_src = r#"
    <- "math/filters.stl" as flt;

    +> sensor;
    => out_signal;

    lp := flt::Lowpass(sensor, out_signal);
    "#;

    let bump = Bump::new();
    let (program, layout) = parse_synaptic_with_resolver(main_src, &resolver, &bump)
        .expect("Aliased module import should resolve and monomorphize cleanly");

    let codegen = stella_compiler::Codegen::new(program, layout);
    let unit = codegen
        .compile()
        .expect("Codegen should succeed for imported module");
    assert!(unit.layout.symbols.contains_key("sensor"));
    assert!(unit.layout.symbols.contains_key("out_signal"));
}

#[test]
fn test_compiler_emit_and_return_signals() {
    let script = r#"
    in sensor;
    out alert;

    sensor ~> alert;

    emit "Sensor reached threshold: {:.2}"(sensor) when sensor >= 0.5;
    emit "System stable." on stable;

    return alert;
    "#;

    let artifact = compile(script, false, None, None).expect("Emit and return should compile");
    assert_eq!(artifact.emits.len(), 2);
    assert_eq!(
        artifact.emits[0].template,
        "Sensor reached threshold: {:.2}"
    );
    assert_eq!(artifact.emits[0].arg_indices.len(), 1);
    assert!(!artifact.emits[0].on_stable);
    assert_eq!(artifact.emits[1].template, "System stable.");
    assert!(artifact.emits[1].on_stable);

    assert!(artifact.return_neuron.is_some());
}
