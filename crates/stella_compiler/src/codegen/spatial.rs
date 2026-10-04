use alloc::format;
use alloc::string::String;
use stella_frontend::ast::{Expr, NodeTarget, PaddingMode};

use super::Codegen;

impl<'a> Codegen<'a> {
    pub(super) fn compile_piecewise_curve(
        &mut self,
        dest: usize,
        inner_expr: &Expr<'a>,
        points: &[(f64, f64)],
    ) -> Result<(), String> {
        if points.is_empty() {
            return Ok(());
        }
        if points.len() == 1 {
            self.add_b(dest, points[0].1);
            return Ok(());
        }

        // Sort points by x coordinate
        let mut sorted = points.to_vec();
        sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(core::cmp::Ordering::Equal));

        let x_src = self.resolve_expr_to_neuron(inner_expr)?;

        let (mut prev_x, mut prev_y) = sorted[0];
        let mut prev_slope = 0.0;

        self.add_b(dest, prev_y);

        for i in 0..(sorted.len() - 1) {
            let (next_x, next_y) = sorted[i + 1];
            let dx = next_x - prev_x;
            let cur_slope = if dx.abs() > 1e-9 {
                (next_y - prev_y) / dx
            } else {
                0.0
            };
            let delta_slope = cur_slope - prev_slope;

            if delta_slope.abs() > 1e-9 {
                let relu_scratch = self.alloc_scratch();
                self.add_w(relu_scratch, x_src, 1.0);
                self.add_b(relu_scratch, -prev_x);

                self.add_w(dest, relu_scratch, delta_slope);
            }

            prev_x = next_x;
            prev_y = next_y;
            prev_slope = cur_slope;
        }

        let final_delta = -prev_slope;
        if final_delta.abs() > 1e-9 {
            let relu_end = self.alloc_scratch();
            self.add_w(relu_end, x_src, 1.0);
            self.add_b(relu_end, -prev_x);
            self.add_w(dest, relu_end, final_delta);
        }

        Ok(())
    }

    pub(super) fn compile_conv2d(
        &mut self,
        dest: &NodeTarget<'a>,
        input_expr: &Expr<'a>,
        kernel_name: &str,
        stride: usize,
        padding: PaddingMode,
        gate: Option<usize>,
    ) -> Result<(), String> {
        let input_name = match input_expr {
            Expr::Ident(name) => *name,
            _ => {
                return Err(
                    "conv2d input currently expects an identifier (e.g. conv2d(retina, K))".into(),
                )
            }
        };

        let in_info = self
            .layout
            .symbols
            .get(input_name)
            .ok_or_else(|| format!("Unknown symbol '{}' in conv2d", input_name))?;

        let (in_rows, in_cols) = match &in_info.shape {
            Some(s) if s.len() == 2 => (s[0], s[1]),
            _ => {
                return Err(format!(
                    "Input '{}' to conv2d must be a 2D tensor (tensor[H, W])",
                    input_name
                ))
            }
        };

        let dest_info = self
            .layout
            .symbols
            .get(dest.name)
            .ok_or_else(|| format!("Unknown destination symbol '{}' in conv2d", dest.name))?;

        let (dest_rows, dest_cols) = match &dest_info.shape {
            Some(s) if s.len() == 2 => (s[0], s[1]),
            _ => {
                return Err(format!(
                    "Destination '{}' for conv2d must be a 2D tensor (tensor[H, W])",
                    dest.name
                ))
            }
        };

        let (k_rows, k_cols, ref k_data) = self
            .layout
            .matrix_constants
            .get(kernel_name)
            .ok_or_else(|| {
                format!(
                    "Unknown 2D matrix constant '{}' used as conv2d kernel",
                    kernel_name
                )
            })?
            .clone();

        let s = if stride == 0 { 1 } else { stride };
        let (expected_out_rows, expected_out_cols, pad_top, pad_left) = match padding {
            PaddingMode::Valid => {
                let r = (in_rows - k_rows) / s + 1;
                let c = (in_cols - k_cols) / s + 1;
                (r, c, 0isize, 0isize)
            }
            PaddingMode::Same => {
                let r = (in_rows + s - 1) / s;
                let c = (in_cols + s - 1) / s;
                let pt = ((r - 1) * s + k_rows).saturating_sub(in_rows) / 2;
                let pl = ((c - 1) * s + k_cols).saturating_sub(in_cols) / 2;
                (r, c, pt as isize, pl as isize)
            }
        };

        if dest_rows != expected_out_rows || dest_cols != expected_out_cols {
            return Err(format!(
                "conv2d output shape mismatch for '{}': expected [{}, {}], but destination is [{}, {}]",
                dest.name, expected_out_rows, expected_out_cols, dest_rows, dest_cols
            ));
        }

        let in_base = in_info.index;
        let dest_base = dest_info.index;

        for out_r in 0..dest_rows {
            for out_c in 0..dest_cols {
                let dest_idx = dest_base + out_r * dest_cols + out_c;
                self.ensure_cleared(dest_idx);

                let in_start_r = (out_r * s) as isize - pad_top;
                let in_start_c = (out_c * s) as isize - pad_left;

                for kr in 0..k_rows {
                    for kc in 0..k_cols {
                        let r = in_start_r + kr as isize;
                        let c = in_start_c + kc as isize;

                        if r >= 0 && r < in_rows as isize && c >= 0 && c < in_cols as isize {
                            let weight_val = k_data[kr * k_cols + kc];
                            if weight_val.abs() > 1e-9 {
                                let src_idx = in_base + (r as usize) * in_cols + (c as usize);
                                if let Some(g) = gate {
                                    let gated_wire = self.alloc_scratch();
                                    self.add_w(gated_wire, g, 1.0);
                                    self.add_w(gated_wire, src_idx, 1.0);
                                    self.add_b(gated_wire, -1.0);
                                    self.add_w(dest_idx, gated_wire, weight_val);
                                } else {
                                    self.add_w(dest_idx, src_idx, weight_val);
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    pub(super) fn compile_avgpool2d(
        &mut self,
        dest: &NodeTarget<'a>,
        input_expr: &Expr<'a>,
        kernel_size: usize,
        stride: usize,
        gate: Option<usize>,
    ) -> Result<(), String> {
        let input_name = match input_expr {
            Expr::Ident(name) => *name,
            _ => return Err("avgpool2d input expects an identifier".into()),
        };

        let in_info = self
            .layout
            .symbols
            .get(input_name)
            .ok_or_else(|| format!("Unknown symbol '{}' in avgpool2d", input_name))?;

        let (in_rows, in_cols) = match &in_info.shape {
            Some(s) if s.len() == 2 => (s[0], s[1]),
            _ => {
                return Err(format!(
                    "Input '{}' to avgpool2d must be a 2D tensor (tensor[H, W])",
                    input_name
                ))
            }
        };

        let dest_info =
            self.layout.symbols.get(dest.name).ok_or_else(|| {
                format!("Unknown destination symbol '{}' in avgpool2d", dest.name)
            })?;

        let (dest_rows, dest_cols) = match &dest_info.shape {
            Some(s) if s.len() == 2 => (s[0], s[1]),
            _ => {
                return Err(format!(
                    "Destination '{}' for avgpool2d must be a 2D tensor (tensor[H, W])",
                    dest.name
                ))
            }
        };

        let k = if kernel_size == 0 { 1 } else { kernel_size };
        let s = if stride == 0 { 1 } else { stride };

        let expected_out_rows = (in_rows - k) / s + 1;
        let expected_out_cols = (in_cols - k) / s + 1;

        if dest_rows != expected_out_rows || dest_cols != expected_out_cols {
            return Err(format!(
                "avgpool2d output shape mismatch for '{}': expected [{}, {}], but destination is [{}, {}]",
                dest.name, expected_out_rows, expected_out_cols, dest_rows, dest_cols
            ));
        }

        let in_base = in_info.index;
        let dest_base = dest_info.index;
        let pool_weight = 1.0 / ((k * k) as f64);

        for out_r in 0..dest_rows {
            for out_c in 0..dest_cols {
                let dest_idx = dest_base + out_r * dest_cols + out_c;
                self.ensure_cleared(dest_idx);

                let in_start_r = out_r * s;
                let in_start_c = out_c * s;

                for kr in 0..k {
                    for kc in 0..k {
                        let src_idx = in_base + (in_start_r + kr) * in_cols + (in_start_c + kc);
                        if let Some(g) = gate {
                            let gated_wire = self.alloc_scratch();
                            self.add_w(gated_wire, g, 1.0);
                            self.add_w(gated_wire, src_idx, 1.0);
                            self.add_b(gated_wire, -1.0);
                            self.add_w(dest_idx, gated_wire, pool_weight);
                        } else {
                            self.add_w(dest_idx, src_idx, pool_weight);
                        }
                    }
                }
            }
        }

        Ok(())
    }

    pub(super) fn compile_matmul(
        &mut self,
        dest: &NodeTarget<'a>,
        input_name: &str,
        matrix_name: &str,
        gate: Option<usize>,
    ) -> Result<(), String> {
        let in_info = self
            .layout
            .symbols
            .get(input_name)
            .ok_or_else(|| format!("Unknown symbol '{}' in @* matmul", input_name))?;

        let dest_info =
            self.layout.symbols.get(dest.name).ok_or_else(|| {
                format!("Unknown destination symbol '{}' in @* matmul", dest.name)
            })?;

        let (m_rows, m_cols, ref m_data) = self
            .layout
            .matrix_constants
            .get(matrix_name)
            .ok_or_else(|| {
                format!(
                    "Unknown matrix constant '{}' used in @* matmul",
                    matrix_name
                )
            })?
            .clone();

        let in_len = in_info.width;
        let out_len = dest_info.width;

        if in_len != m_rows || out_len != m_cols {
            return Err(format!(
                "Dimension mismatch in @* matmul: vector[{}] @* matrix[{}, {}] -> destination[{}]",
                in_len, m_rows, m_cols, out_len
            ));
        }

        let in_base = in_info.index;
        let dest_base = dest_info.index;

        for c in 0..m_cols {
            let d_idx = dest_base + c;
            self.ensure_cleared(d_idx);
            for r in 0..m_rows {
                let weight = m_data[r * m_cols + c];
                if weight.abs() > 1e-9 {
                    let s_idx = in_base + r;
                    if let Some(g) = gate {
                        let gated_wire = self.alloc_scratch();
                        self.add_w(gated_wire, g, 1.0);
                        self.add_w(gated_wire, s_idx, 1.0);
                        self.add_b(gated_wire, -1.0);
                        self.add_w(d_idx, gated_wire, weight);
                    } else {
                        self.add_w(d_idx, s_idx, weight);
                    }
                }
            }
        }

        Ok(())
    }

    pub(super) fn compile_outer_product(
        &mut self,
        dest: &NodeTarget<'a>,
        left_name: &str,
        right_name: &str,
        gate: Option<usize>,
    ) -> Result<(), String> {
        let left_info = self
            .layout
            .symbols
            .get(left_name)
            .ok_or_else(|| format!("Unknown symbol '{}' in ^* outer product", left_name))?;

        let right_info = self
            .layout
            .symbols
            .get(right_name)
            .ok_or_else(|| format!("Unknown symbol '{}' in ^* outer product", right_name))?;

        let dest_info = self.layout.symbols.get(dest.name).ok_or_else(|| {
            format!(
                "Unknown destination symbol '{}' in ^* outer product",
                dest.name
            )
        })?;

        let (dest_rows, dest_cols) = match &dest_info.shape {
            Some(s) if s.len() == 2 => (s[0], s[1]),
            _ => (dest_info.width, 1),
        };

        let l_len = left_info.width;
        let r_len = right_info.width;

        if dest_rows != l_len || dest_cols != r_len {
            if dest_info.width != l_len * r_len {
                return Err(format!(
                    "Dimension mismatch in ^* outer product: left[{}] ^* right[{}] -> dest[{}, {}] (width {})",
                    l_len, r_len, dest_rows, dest_cols, dest_info.width
                ));
            }
        }

        let l_base = left_info.index;
        let r_base = right_info.index;
        let dest_base = dest_info.index;

        // dest[i, j] = left[i] * right[j] via multiplication gadget
        for i in 0..l_len {
            for j in 0..r_len {
                let d_idx = dest_base + i * r_len + j;
                self.ensure_cleared(d_idx);

                let mult_wire = self.alloc_scratch();
                let s_left = l_base + i;
                let s_right = r_base + j;

                // Multiplication gadget: min(left, right) or gated correlation
                // Wire: (left + right - 1.0) with threshold/offset
                self.add_w(mult_wire, s_left, 1.0);
                self.add_w(mult_wire, s_right, 1.0);
                self.add_b(mult_wire, -1.0);

                if let Some(g) = gate {
                    let gated_wire = self.alloc_scratch();
                    self.add_w(gated_wire, g, 1.0);
                    self.add_w(gated_wire, mult_wire, 1.0);
                    self.add_b(gated_wire, -1.0);
                    self.add_w(d_idx, gated_wire, 1.0);
                } else {
                    self.add_w(d_idx, mult_wire, 1.0);
                }
            }
        }

        Ok(())
    }

    pub(super) fn compile_energy_minimize(
        &mut self,
        energy_expr: &Expr<'a>,
        gate: Option<usize>,
    ) -> Result<(), String> {
        // Continuous gradient flow: dx/dt = -grad E(x)
        // If E = sum (x_i - target_i)^2 or (x - y)^2, synthesize gradient descent weights
        match energy_expr {
            Expr::Binary {
                op: stella_frontend::ast::BinOp::Add,
                left,
                right,
            } => {
                self.compile_energy_minimize(left, gate)?;
                self.compile_energy_minimize(right, gate)?;
                Ok(())
            }
            Expr::Binary {
                op: stella_frontend::ast::BinOp::Mul,
                left,
                right,
            } => {
                // Quadratic well: (x - c)^2 or (x - y)^2
                if left == right {
                    self.compile_quadratic_well(left, gate)
                } else {
                    let energy_neuron = self.resolve_expr_to_neuron(energy_expr)?;
                    if let Some(g) = gate {
                        self.add_w(energy_neuron, g, -0.5);
                    }
                    Ok(())
                }
            }
            _ => {
                let energy_neuron = self.resolve_expr_to_neuron(energy_expr)?;
                if let Some(g) = gate {
                    self.add_w(energy_neuron, g, -0.5);
                }
                Ok(())
            }
        }
    }

    fn compile_quadratic_well(
        &mut self,
        diff_expr: &Expr<'a>,
        gate: Option<usize>,
    ) -> Result<(), String> {
        match diff_expr {
            Expr::Binary {
                op: stella_frontend::ast::BinOp::Sub,
                left,
                right,
            } => {
                let left_target = match left {
                    Expr::Ident(name) => self.layout.resolve_ident(name).ok(),
                    _ => None,
                };
                let right_target = match right {
                    Expr::Ident(name) => self.layout.resolve_ident(name).ok(),
                    _ => None,
                };

                match (left_target, right_target) {
                    (Some(l_idx), Some(r_idx)) => {
                        // E = (x - y)^2 => dx/dt = -2(x - y), dy/dt = -2(y - x) = 2(x - y)
                        // Gradient pulls x toward y, y toward x
                        let rate = 0.5;
                        if let Some(g) = gate {
                            let wire_l = self.alloc_scratch();
                            self.add_w(wire_l, g, 1.0);
                            self.add_w(wire_l, r_idx, 1.0);
                            self.add_b(wire_l, -1.0);
                            self.add_w(l_idx, wire_l, rate);

                            let wire_r = self.alloc_scratch();
                            self.add_w(wire_r, g, 1.0);
                            self.add_w(wire_r, l_idx, 1.0);
                            self.add_b(wire_r, -1.0);
                            self.add_w(r_idx, wire_r, rate);

                            // Also register direct attractor basin coupling
                            self.add_w(l_idx, r_idx, rate);
                            self.add_w(r_idx, l_idx, rate);
                        } else {
                            self.add_w(l_idx, r_idx, rate);
                            self.add_w(r_idx, l_idx, rate);
                        }
                        Ok(())
                    }
                    (Some(l_idx), None) => {
                        if let Some(val) = self.eval_const(right) {
                            let rate = 0.5;
                            if let Some(g) = gate {
                                self.add_w(l_idx, g, rate * val);
                            } else {
                                self.add_b(l_idx, rate * val);
                            }
                        }
                        Ok(())
                    }
                    _ => Ok(()),
                }
            }
            _ => Ok(()),
        }
    }
}
