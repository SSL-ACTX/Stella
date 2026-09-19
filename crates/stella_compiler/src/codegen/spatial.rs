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
}
