use alloc::format;
use alloc::string::String;
use stella_frontend::ast::Expr;

use super::Codegen;

impl<'a> Codegen<'a> {
    pub(super) fn compile_dynamic_read(
        &mut self,
        dest: usize,
        bank_name: &str,
        addr_name: &str,
    ) -> Result<(), String> {
        let (bank_base, bank_len) = {
            let info = self
                .layout
                .symbols
                .get(bank_name)
                .ok_or_else(|| format!("Unknown symbol '{}'", bank_name))?;
            (info.index, info.width)
        };
        let (addr_base, addr_len) = {
            let info = self
                .layout
                .symbols
                .get(addr_name)
                .ok_or_else(|| format!("Unknown symbol '{}'", addr_name))?;
            (info.index, info.width)
        };

        if addr_len == 1 {
            for i in 0..bank_len {
                let diff = self.alloc_scratch();
                self.add_w(diff, addr_base, 1.0);
                self.add_b(diff, -(i as f64));

                let up = self.alloc_scratch();
                self.add_w(up, diff, 1.0);
                self.add_b(up, 1.0);

                let down = self.alloc_scratch();
                self.add_w(down, diff, -1.0);
                self.add_b(down, 1.0);

                let sel_i = self.alloc_scratch();
                self.add_w(sel_i, up, 1.0);
                self.add_w(sel_i, down, 1.0);
                self.add_b(sel_i, -1.0);

                let prod_i = self.alloc_scratch();
                self.add_w(prod_i, sel_i, 1.0);
                self.add_w(prod_i, bank_base + i, 1.0);
                self.add_b(prod_i, -1.0);

                self.add_w(dest, prod_i, 1.0);
            }
        } else {
            let max_addrs = 1usize << addr_len;
            let count = bank_len.min(max_addrs);
            for i in 0..count {
                let mut num_ones = 0.0;
                let dec_i = self.alloc_scratch();
                for bit in 0..addr_len {
                    if (i & (1 << bit)) != 0 {
                        self.add_w(dec_i, addr_base + bit, 1.0);
                        num_ones += 1.0;
                    } else {
                        self.add_w(dec_i, addr_base + bit, -1.0);
                    }
                }
                self.add_b(dec_i, -(num_ones - 0.5));

                let sel_i = self.alloc_scratch();
                self.add_w(sel_i, dec_i, 10.0);
                self.add_b(sel_i, -4.0);

                let prod_i = self.alloc_scratch();
                self.add_w(prod_i, sel_i, 1.0);
                self.add_w(prod_i, bank_base + i, 1.0);
                self.add_b(prod_i, -1.0);

                self.add_w(dest, prod_i, 1.0);
            }
        }
        Ok(())
    }

    pub(super) fn compile_dynamic_write(
        &mut self,
        bank_name: &str,
        addr_name: &str,
        expr: &Expr<'a>,
        gate: Option<usize>,
    ) -> Result<(), String> {
        let (bank_base, bank_len) = {
            let info = self
                .layout
                .symbols
                .get(bank_name)
                .ok_or_else(|| format!("Unknown symbol '{}'", bank_name))?;
            (info.index, info.width)
        };
        let (addr_base, addr_len) = {
            let info = self
                .layout
                .symbols
                .get(addr_name)
                .ok_or_else(|| format!("Unknown symbol '{}'", addr_name))?;
            (info.index, info.width)
        };

        let val_src = self.resolve_expr_to_neuron(expr)?;

        let num_addrs = if addr_len == 1 {
            bank_len
        } else {
            bank_len.min(1usize << addr_len)
        };

        for i in 0..num_addrs {
            let sel_i = if addr_len == 1 {
                let diff = self.alloc_scratch();
                self.add_w(diff, addr_base, 1.0);
                self.add_b(diff, -(i as f64));

                let up = self.alloc_scratch();
                self.add_w(up, diff, 1.0);
                self.add_b(up, 1.0);

                let down = self.alloc_scratch();
                self.add_w(down, diff, -1.0);
                self.add_b(down, 1.0);

                let s = self.alloc_scratch();
                self.add_w(s, up, 1.0);
                self.add_w(s, down, 1.0);
                self.add_b(s, -1.0);
                s
            } else {
                let mut num_ones = 0.0;
                let dec_i = self.alloc_scratch();
                for bit in 0..addr_len {
                    if (i & (1 << bit)) != 0 {
                        self.add_w(dec_i, addr_base + bit, 1.0);
                        num_ones += 1.0;
                    } else {
                        self.add_w(dec_i, addr_base + bit, -1.0);
                    }
                }
                self.add_b(dec_i, -(num_ones - 0.5));
                let s = self.alloc_scratch();
                self.add_w(s, dec_i, 10.0);
                self.add_b(s, -4.0);
                s
            };

            let eff_sel = if let Some(g) = gate {
                let combined = self.alloc_scratch();
                self.add_w(combined, g, 1.0);
                self.add_w(combined, sel_i, 1.0);
                self.add_b(combined, -1.0);
                combined
            } else {
                sel_i
            };

            let cell = bank_base + i;
            let term1 = self.alloc_scratch();
            self.add_w(term1, cell, 1.0);
            self.add_w(term1, eff_sel, -1.0);

            let term2 = self.alloc_scratch();
            self.add_w(term2, val_src, 1.0);
            self.add_w(term2, eff_sel, 1.0);
            self.add_b(term2, -1.0);

            self.ensure_cleared(cell);
            self.add_w(cell, term1, 1.0);
            self.add_w(cell, term2, 1.0);
        }
        Ok(())
    }
}
