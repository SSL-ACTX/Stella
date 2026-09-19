// crates/stella_frontend/src/ast.rs

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Program<'a> {
    pub declarations: &'a [Decl<'a>],
    pub flows: &'a [Flow<'a>],
    pub basins: &'a [Basin<'a>],
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NodeTarget<'a> {
    pub name: &'a str,
    pub index: Option<usize>,
    pub indices: Option<&'a [usize]>,
    pub dynamic_index: Option<&'a str>,
    pub slice: Option<(usize, usize)>,
}

impl<'a> NodeTarget<'a> {
    pub fn simple(name: &'a str) -> Self {
        Self {
            name,
            index: None,
            indices: None,
            dynamic_index: None,
            slice: None,
        }
    }

    pub fn dynamic(name: &'a str, addr: &'a str) -> Self {
        Self {
            name,
            index: None,
            indices: None,
            dynamic_index: Some(addr),
            slice: None,
        }
    }

    pub fn indexed(name: &'a str, index: usize) -> Self {
        Self {
            name,
            index: Some(index),
            indices: None,
            dynamic_index: None,
            slice: None,
        }
    }

    pub fn multidim(name: &'a str, indices: &'a [usize]) -> Self {
        Self {
            name,
            index: None,
            indices: Some(indices),
            dynamic_index: None,
            slice: None,
        }
    }

    pub fn sliced(name: &'a str, start: usize, end: usize) -> Self {
        Self {
            name,
            index: None,
            indices: None,
            dynamic_index: None,
            slice: Some((start, end)),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Decl<'a> {
    TerminalIn {
        name: &'a str,
        width: usize,
        shape: Option<&'a [usize]>,
    },
    TerminalOut {
        name: &'a str,
        width: usize,
        shape: Option<&'a [usize]>,
    },
    Node {
        name: &'a str,
        width: usize,
        shape: Option<&'a [usize]>,
        dynamics: NodeDynamics,
        initial: Option<f64>,
    },
    Const(&'a str, f64),
    ConstMatrix {
        name: &'a str,
        rows: usize,
        cols: usize,
        data: &'a [f64],
    },
    /// In-source chaotic obfuscation configuration directive: `cloak { pad <size>; seed <x>, <y>; }`
    Cloak {
        pad: usize,
        seed: Option<(f64, f64)>,
    },
    /// Reusable synaptic circuit subgraph definition: `circuit Name<templates>(params) -> return_param { ... }`
    Circuit {
        name: &'a str,
        template_params: &'a [TemplateParam<'a>],
        params: &'a [CircuitParam<'a>],
        return_param: Option<CircuitParam<'a>>,
        declarations: &'a [Decl<'a>],
        flows: &'a [Flow<'a>],
    },
    /// Circuit instantiation: `inst instance_name = CircuitName<templates>(arg1, arg2, ...);`
    Inst {
        name: &'a str,
        circuit: &'a str,
        template_args: &'a [f64],
        args: &'a [NodeTarget<'a>],
    },
    /// Compile-time spectral stability assertion: `assert stable;` or `assert stable(loop_name);`
    AssertStable {
        target: Option<&'a str>,
    },
    /// Dynamic bounded state assertion: `assert bounded(signal, min, max);`
    AssertBounded {
        target: NodeTarget<'a>,
        min: f64,
        max: f64,
    },
    /// Universal shape type alias: `type uint8 = [8];` or `type matrix = [4, 4];`
    TypeAlias {
        name: &'a str,
        shape: &'a [usize],
    },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TemplateParam<'a> {
    pub name: &'a str,
    pub default: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamDirection {
    In,
    Out,
    InOut,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CircuitParam<'a> {
    pub direction: ParamDirection,
    pub name: &'a str,
    pub width: usize,
    pub shape: Option<&'a [usize]>,
    pub default: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NodeDynamics {
    Standard,
    Leak(f64),
    Latch,
    Hold,
    Plastic { rate: f64, decay: f64 },
    Oscillator { period: usize },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Flow<'a> {
    /// Synaptic excitation `src ~> dest` or `src ~[weight]> dest` (with optional `when <cond>`)
    Synapse {
        src: NodeTarget<'a>,
        dest: NodeTarget<'a>,
        weight: f64,
        when: Option<&'a Expr<'a>>,
    },
    /// Synaptic inhibition `src ~|> dest` (-1.0 drain)
    Inhibit {
        src: NodeTarget<'a>,
        dest: NodeTarget<'a>,
        when: Option<&'a Expr<'a>>,
    },
    /// Synaptic broadcasting `src ~> [dest1, dest2, ...]`
    Broadcast {
        src: NodeTarget<'a>,
        dests: &'a [NodeTarget<'a>],
        weight: f64,
        when: Option<&'a Expr<'a>>,
    },
    /// Synaptic fan-in / summing `[src1, src2, ...] ~> dest`
    FanIn {
        srcs: &'a [NodeTarget<'a>],
        dest: NodeTarget<'a>,
        weight: f64,
        when: Option<&'a Expr<'a>>,
    },
    /// Synaptic broadcast inhibition `src ~|> [dest1, dest2, ...]`
    BroadcastInhibit {
        src: NodeTarget<'a>,
        dests: &'a [NodeTarget<'a>],
        when: Option<&'a Expr<'a>>,
    },
    /// Synaptic fan-in inhibition `[src1, src2, ...] ~|> dest`
    FanInInhibit {
        srcs: &'a [NodeTarget<'a>],
        dest: NodeTarget<'a>,
        when: Option<&'a Expr<'a>>,
    },
    /// Algebraic expression assignment `dest = expr` or `dest += expr`
    Assign {
        dest: NodeTarget<'a>,
        expr: &'a Expr<'a>,
    },
    /// Zero out recurrent state `drain dest`
    Drain(NodeTarget<'a>),
    /// Dynamic Hebbian/plasticity rule: `learn pre ~> dest : rate [, decay];` or `synapse pre ~> dest { plastic: true, ... }`
    Plastic {
        pre: NodeTarget<'a>,
        dest: NodeTarget<'a>,
        rate: f64,
        decay: f64,
        kind: &'a str,
    },
    /// Conditional branching CFG: `if cond { ... } else { ... }`
    If {
        cond: &'a Expr<'a>,
        then_flows: &'a [Flow<'a>],
        else_flows: Option<&'a [Flow<'a>]>,
    },
    /// Recurrent attractor loop CFG: `while cond { ... }`
    While {
        cond: &'a Expr<'a>,
        body: &'a [Flow<'a>],
    },
    /// Observable telemetry probe / print statement: `probe "label", expr;` or `probe expr;`
    Probe {
        label: Option<&'a str>,
        expr: &'a Expr<'a>,
    },
    /// Inline gated synaptic flow: `src ~[weight]> (gate: cond) ~> dest`
    GatedSynapse {
        src: NodeTarget<'a>,
        dest: NodeTarget<'a>,
        gate: &'a Expr<'a>,
        weight: f64,
    },
    /// Presynaptic shunted flow: `src ~[weight]> [shunt: cond] ~> dest`
    ShuntedSynapse {
        src: NodeTarget<'a>,
        dest: NodeTarget<'a>,
        shunt: &'a Expr<'a>,
        weight: f64,
    },
    /// Synaptic broadcast gated flow: `src ~> (gate: cond) ~> [d1, d2, ...]`
    BroadcastGated {
        src: NodeTarget<'a>,
        dests: &'a [NodeTarget<'a>],
        gate: &'a Expr<'a>,
        weight: f64,
    },
    /// Synaptic broadcast shunted flow: `src ~> [shunt: cond] ~> [d1, d2, ...]`
    BroadcastShunted {
        src: NodeTarget<'a>,
        dests: &'a [NodeTarget<'a>],
        shunt: &'a Expr<'a>,
        weight: f64,
    },
    /// Synaptic fan-in gated flow: `[s1, s2, ...] ~> (gate: cond) ~> dest`
    FanInGated {
        srcs: &'a [NodeTarget<'a>],
        dest: NodeTarget<'a>,
        gate: &'a Expr<'a>,
        weight: f64,
    },
    /// Synaptic fan-in shunted flow: `[s1, s2, ...] ~> [shunt: cond] ~> dest`
    FanInShunted {
        srcs: &'a [NodeTarget<'a>],
        dest: NodeTarget<'a>,
        shunt: &'a Expr<'a>,
        weight: f64,
    },
    /// Continuous bifurcation routing: `bifurcate (expr) { branches }`
    Bifurcate {
        expr: &'a Expr<'a>,
        branches: &'a [BifurcateBranch<'a>],
    },
    /// Competitive lateral inhibition: `compete { branches } resolve winner_take_all(threshold);`
    Compete {
        branches: &'a [CompeteBranch<'a>],
        threshold: f64,
    },
    /// Continuous relaxation settling loop: `relax { body } until stable(tolerance) [or timeout(cycles)];`
    Relax {
        body: &'a [Flow<'a>],
        tolerance: f64,
        timeout: Option<usize>,
    },
    /// Concurrent state superposition & collapse: `superpose { branches } collapse_on expr ~> dest;`
    Superpose {
        branches: &'a [SuperposeBranch<'a>],
        collapse_expr: &'a Expr<'a>,
        dest: NodeTarget<'a>,
    },
    /// Multi-branch compound flow fan-out: `src ~> ( |> filter ~> dest1, * weight ~> dest2, ... );`
    MultiBranch {
        src: NodeTarget<'a>,
        branches: &'a [BranchPipeline<'a>],
    },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PipelineStep {
    Filter(ActivationKind),
    Scale(f64),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BranchPipeline<'a> {
    pub steps: &'a [PipelineStep],
    pub dests: &'a [NodeTarget<'a>],
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BifurcateBranch<'a> {
    pub cond: BifurcateCond<'a>,
    pub target: Option<NodeTarget<'a>>,
    pub flows: &'a [Flow<'a>],
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BifurcateCond<'a> {
    Lt(f64),
    Lte(f64),
    Gt(f64),
    Gte(f64),
    Range(f64, f64),
    Eq(f64),
    When(&'a Expr<'a>),
    Else,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CompeteBranch<'a> {
    pub name: &'a str,
    pub flows: &'a [Flow<'a>],
    pub head: Option<NodeTarget<'a>>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SuperposeBranch<'a> {
    pub name: &'a str,
    pub flows: &'a [Flow<'a>],
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Basin<'a> {
    pub name: &'a str,
    pub flows: &'a [Flow<'a>],
    pub bifurcations: &'a [Bifurcation<'a>],
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bifurcation<'a> {
    pub target: &'a str,
    pub condition: Option<&'a Expr<'a>>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Expr<'a> {
    Number(f64),
    Ident(&'a str),
    Index {
        name: &'a str,
        index: usize,
    },
    MultiIndex {
        name: &'a str,
        indices: &'a [usize],
    },
    DynamicIndex {
        name: &'a str,
        addr: &'a str,
    },
    Binary {
        op: BinOp,
        left: &'a Expr<'a>,
        right: &'a Expr<'a>,
    },
    Unary {
        op: UnaryOp,
        inner: &'a Expr<'a>,
    },
    Activation {
        kind: ActivationKind,
        expr: &'a Expr<'a>,
    },
    CircuitCall {
        circuit: &'a str,
        template_args: &'a [f64],
        args: &'a [&'a Expr<'a>],
    },
    Curve {
        expr: &'a Expr<'a>,
        points: &'a [(f64, f64)],
    },
    Match {
        expr: &'a Expr<'a>,
        pattern: &'a str,
    },
    Conv2d {
        input: &'a Expr<'a>,
        kernel: &'a str,
        stride: usize,
        padding: PaddingMode,
    },
    AvgPool2d {
        input: &'a Expr<'a>,
        kernel_size: usize,
        stride: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaddingMode {
    Valid,
    Same,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Neg,
    Not,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationKind {
    Saturate, // High-gain op-amp saturating step function
    Clamp,    // Clamped linear range [0.0, 1.0]
    Step,     // Heaviside step at threshold 0.5
    Inv,      // Linear signal inversion: 1.0 - x
    Relu,     // Rectified linear unit: max(0, x)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    And,
    Or,
    Nand,
    Nor,
    Xor,
    Gte,
    Lte,
    Gt,
    Lt,
    Eq,
    Neq,
}
