// crates/stella_frontend/src/parser/handlers/decls.rs
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;
use pest::iterators::Pair;

use crate::ast::*;
use crate::parser::{Lowerer, Rule};

impl<'a> Lowerer<'a> {
    pub fn lower_attribute(&mut self, pair: Pair<'a, Rule>) -> Result<Option<Decl<'a>>, String> {
        let text = pair.as_str().trim();
        if text.starts_with("^|") {
            // Range bound contract: ^|ident| <= number;
            let second_bar = text[2..].find('|').unwrap() + 2;
            let target_name = text[2..second_bar].trim();
            let rest = text[second_bar + 1..].trim().trim_end_matches(';').trim();
            let max_val = if rest.starts_with("<=") {
                rest[2..].trim().parse::<f64>().unwrap_or(1.0)
            } else if rest.starts_with('<') {
                rest[1..].trim().parse::<f64>().unwrap_or(1.0)
            } else {
                1.0
            };
            return Ok(Some(Decl::AssertBounded {
                target: NodeTarget::simple(target_name),
                min: -max_val,
                max: max_val,
            }));
        }

        let mut inner = pair.into_inner();
        let name = inner.next().unwrap().as_str();

        match name {
            "stable" => {
                let target = inner.next().and_then(|args| {
                    args.into_inner()
                        .next()
                        .map(|arg| arg.as_str().trim_matches('"'))
                });
                Ok(Some(Decl::AssertStable { target }))
            }
            "bounded" => {
                let args_pair = inner
                    .next()
                    .ok_or_else(|| "Attribute #[bounded] requires arguments".to_string())?;
                let mut args = args_pair.into_inner();
                let target_pair = args
                    .next()
                    .ok_or_else(|| "Missing target in #[bounded]".to_string())?;
                let target = NodeTarget::simple(target_pair.as_str());
                let min = args
                    .next()
                    .ok_or_else(|| "Missing min in #[bounded]".to_string())?
                    .as_str()
                    .parse::<f64>()
                    .map_err(|e| format!("Invalid min: {}", e))?;
                let max = args
                    .next()
                    .ok_or_else(|| "Missing max in #[bounded]".to_string())?
                    .as_str()
                    .parse::<f64>()
                    .map_err(|e| format!("Invalid max: {}", e))?;

                Ok(Some(Decl::AssertBounded { target, min, max }))
            }
            "cloak" => {
                let mut pad = 0;
                let mut seed = None;

                if let Some(args_pair) = inner.next() {
                    for arg in args_pair.into_inner() {
                        let arg_inner_pair = arg.into_inner().next().unwrap();
                        if arg_inner_pair.as_rule() == Rule::attr_named {
                            let mut named_inner = arg_inner_pair.into_inner();
                            let key = named_inner.next().unwrap().as_str().trim();
                            let val_pair = named_inner.next().unwrap();
                            let val_str = val_pair.as_str().trim();
                            if key == "pad" {
                                pad = val_str.parse::<usize>().unwrap_or(0);
                            } else if key == "seed" {
                                if let Ok((sx, sy)) = self.parse_seed_val(val_pair) {
                                    seed = Some((sx, sy));
                                }
                            }
                        }
                    }
                }

                Ok(Some(Decl::Cloak { pad, seed }))
            }
            other => Err(format!("Unknown attribute '#[{}]'", other)),
        }
    }

    fn parse_seed_val(&self, pair: Pair<'a, Rule>) -> Result<(f64, f64), String> {
        let val_pair = pair.into_inner().next().unwrap();
        let mut items = Vec::new();
        for item in val_pair.into_inner() {
            let s = item.as_str().trim();
            if let Ok(num) = s.parse::<f64>() {
                items.push(num);
            }
        }
        if items.len() >= 2 {
            Ok((items[0], items[1]))
        } else {
            Err("Expected two seed values".to_string())
        }
    }

    pub fn lower_decl(&mut self, pair: Pair<'a, Rule>) -> Result<Vec<Decl<'a>>, String> {
        let inner = pair.into_inner().next().unwrap();
        match inner.as_rule() {
            Rule::import_decl => {
                let mut pairs = inner.into_inner();
                let path_pair = pairs.next().unwrap();
                let path = path_pair.as_str().trim_matches('"');
                let target = if let Some(t_pair) = pairs.next() {
                    let mut t_inner = t_pair.into_inner();
                    let first = t_inner.next().unwrap();
                    match first.as_rule() {
                        Rule::ident => ImportTarget::Alias(first.as_str()),
                        Rule::import_selective => {
                            let mut names = Vec::new();
                            for id in first.into_inner() {
                                names.push(id.as_str());
                            }
                            ImportTarget::Selective(self.bump.alloc_slice_copy(&names))
                        }
                        _ => ImportTarget::Open,
                    }
                } else {
                    ImportTarget::Open
                };
                Ok(vec![Decl::Import { path, target }])
            }
            Rule::type_decl => {
                let mut pairs = inner.into_inner();
                let name = pairs.next().unwrap().as_str();
                let shape_pair = pairs.next().unwrap();
                let shape = self.lower_shape(shape_pair)?;
                let width: usize = shape.iter().product();
                let shape_opt = if shape.len() > 1 { Some(shape) } else { None };
                self.type_aliases.insert(name, (width, shape_opt));
                Ok(vec![Decl::TypeAlias { name, shape }])
            }
            Rule::in_decl => {
                let mut decls = Vec::new();
                for item in inner.into_inner() {
                    let (name, width, shape) = self.lower_port_item(item)?;
                    decls.push(Decl::TerminalIn { name, width, shape });
                }
                Ok(decls)
            }
            Rule::out_decl => {
                let mut decls = Vec::new();
                for item in inner.into_inner() {
                    let (name, width, shape) = self.lower_port_item(item)?;
                    decls.push(Decl::TerminalOut { name, width, shape });
                }
                Ok(decls)
            }
            Rule::node_decl => {
                let mut inner_pairs = inner.into_inner();
                let name = inner_pairs.next().unwrap().as_str();

                let mut width = 1;
                let mut shape = None;
                let mut dynamics = NodeDynamics::Standard;
                let mut initial = None;

                for p in inner_pairs {
                    match p.as_rule() {
                        Rule::shape => {
                            let sh = self.lower_shape(p)?;
                            width = sh.iter().product();
                            shape = if sh.len() > 1 { Some(sh) } else { None };
                        }
                        Rule::node_type_or_dyn => {
                            for sub in p.into_inner() {
                                match sub.as_rule() {
                                    Rule::node_dynamics => {
                                        dynamics = self.lower_node_dynamics(sub)?;
                                    }
                                    Rule::type_spec => {
                                        let (w, sh) = self.lower_type_spec(sub)?;
                                        width = w;
                                        shape = sh;
                                    }
                                    _ => {}
                                }
                            }
                        }
                        Rule::node_dynamics => {
                            dynamics = self.lower_node_dynamics(p)?;
                        }
                        Rule::number => {
                            initial = Some(p.as_str().parse::<f64>().map_err(|e| e.to_string())?);
                        }
                        _ => {}
                    }
                }

                Ok(vec![Decl::Node {
                    name,
                    width,
                    shape,
                    dynamics,
                    initial,
                }])
            }
            Rule::const_decl => {
                let mut inner_pairs = inner.into_inner();
                let name = inner_pairs.next().unwrap().as_str();
                let mut val_pair = inner_pairs.next().unwrap();
                if val_pair.as_rule() == Rule::type_spec {
                    val_pair = inner_pairs.next().unwrap();
                }

                let const_val_pair = val_pair.into_inner().next().unwrap();
                match const_val_pair.as_rule() {
                    Rule::number => {
                        let val = const_val_pair
                            .as_str()
                            .parse::<f64>()
                            .map_err(|e| e.to_string())?;
                        Ok(vec![Decl::Const(name, val)])
                    }
                    Rule::matrix_lit => {
                        let mut rows = 0;
                        let mut cols = 0;
                        let mut data = Vec::new();

                        for row_pair in const_val_pair.into_inner() {
                            rows += 1;
                            let mut row_cols = 0;
                            for num_pair in row_pair.into_inner() {
                                row_cols += 1;
                                let num = num_pair
                                    .as_str()
                                    .parse::<f64>()
                                    .map_err(|e| e.to_string())?;
                                data.push(num);
                            }
                            cols = row_cols;
                        }

                        Ok(vec![Decl::ConstMatrix {
                            name,
                            rows,
                            cols,
                            data: self.bump.alloc_slice_copy(&data),
                        }])
                    }
                    other => Err(format!("Unexpected const_val rule: {:?}", other)),
                }
            }
            Rule::circuit_decl => {
                let mut pairs = inner.into_inner();
                let name = pairs.next().unwrap().as_str();

                let mut template_params = Vec::new();
                let mut params = Vec::new();
                let mut return_param = None;
                let mut declarations = Vec::new();
                let mut flows = Vec::new();

                for p in pairs {
                    match p.as_rule() {
                        Rule::template_params => {
                            for t in p.into_inner() {
                                let mut t_inner = t.into_inner();
                                let t_name = t_inner.next().unwrap().as_str();
                                let default = t_inner
                                    .next()
                                    .map(|num| num.as_str().parse::<f64>().unwrap());
                                template_params.push(TemplateParam {
                                    name: t_name,
                                    default,
                                });
                            }
                        }
                        Rule::circuit_params => {
                            for cp in p.into_inner() {
                                let cp_inner = cp.into_inner();
                                let mut direction = ParamDirection::InOut;
                                let mut pname = "";
                                let mut width = 1;
                                let mut shape = None;
                                let mut default = None;

                                for sub in cp_inner {
                                    match sub.as_rule() {
                                        Rule::param_dir => {
                                            direction = match sub.as_str() {
                                                "+>" | "in" => ParamDirection::In,
                                                "=>" | "out" => ParamDirection::Out,
                                                _ => ParamDirection::InOut,
                                            };
                                        }
                                        Rule::shape => {
                                            let sh = self.lower_shape(sub)?;
                                            width = sh.iter().product();
                                            shape = if sh.len() > 1 { Some(sh) } else { None };
                                        }
                                        Rule::ident => {
                                            if pname.is_empty() {
                                                pname = sub.as_str().trim();
                                            }
                                        }
                                        Rule::type_spec => {
                                            let (w, s) = self.lower_type_spec(sub)?;
                                            width = w;
                                            shape = s;
                                        }
                                        Rule::number => {
                                            default = Some(sub.as_str().parse::<f64>().unwrap());
                                        }
                                        _ => {}
                                    }
                                }

                                params.push(CircuitParam {
                                    direction,
                                    name: pname,
                                    width,
                                    shape,
                                    default,
                                });
                            }
                        }
                        Rule::circuit_return => {
                            let mut ret_inner = p.into_inner();
                            let last = ret_inner.next_back().unwrap();
                            let rname = last.as_str().trim();
                            return_param = Some(CircuitParam {
                                direction: ParamDirection::Out,
                                name: rname,
                                width: 1,
                                shape: None,
                                default: None,
                            });
                        }
                        Rule::item => {
                            for sub in p.into_inner() {
                                match sub.as_rule() {
                                    Rule::decl => {
                                        let mut d_list = self.lower_decl(sub)?;
                                        declarations.append(&mut d_list);
                                    }
                                    Rule::flow_stmt => {
                                        let mut f_list = self.lower_flow_stmt(sub)?;
                                        flows.append(&mut f_list);
                                    }
                                    _ => {}
                                }
                            }
                        }
                        Rule::decl => {
                            let mut d_list = self.lower_decl(p)?;
                            declarations.append(&mut d_list);
                        }
                        Rule::flow_stmt => {
                            let mut f_list = self.lower_flow_stmt(p)?;
                            flows.append(&mut f_list);
                        }
                        _ => {}
                    }
                }

                Ok(vec![Decl::Circuit {
                    name,
                    template_params: self.bump.alloc_slice_copy(&template_params),
                    params: self.bump.alloc_slice_copy(&params),
                    return_param,
                    declarations: self.bump.alloc_slice_copy(&declarations),
                    flows: self.bump.alloc_slice_copy(&flows),
                }])
            }
            Rule::inst_decl => {
                let mut pairs = inner.into_inner();
                let name = pairs.next().unwrap().as_str();
                let circuit = pairs.next().unwrap().as_str();

                let mut template_args = Vec::new();
                let mut args = Vec::new();

                for p in pairs {
                    match p.as_rule() {
                        Rule::template_args => {
                            for num in p.into_inner() {
                                template_args.push(num.as_str().parse::<f64>().unwrap());
                            }
                        }
                        Rule::inst_args => {
                            for t in p.into_inner() {
                                args.push(self.lower_node_target(t)?);
                            }
                        }
                        _ => {}
                    }
                }

                Ok(vec![Decl::Inst {
                    name,
                    circuit,
                    template_args: self.bump.alloc_slice_copy(&template_args),
                    args: self.bump.alloc_slice_copy(&args),
                }])
            }
            other => Err(format!("Unhandled declaration rule: {:?}", other)),
        }
    }

    pub fn lower_shape(&self, pair: Pair<'a, Rule>) -> Result<&'a [usize], String> {
        let mut dims = Vec::new();
        for dim in pair.into_inner() {
            let val = dim
                .as_str()
                .parse::<usize>()
                .map_err(|e| format!("Invalid shape dimension: {}", e))?;
            dims.push(val);
        }
        Ok(self.bump.alloc_slice_copy(&dims))
    }

    pub fn lower_type_spec(
        &self,
        pair: Pair<'a, Rule>,
    ) -> Result<(usize, Option<&'a [usize]>), String> {
        let inner = pair.into_inner().next().unwrap();
        match inner.as_rule() {
            Rule::shape => {
                let shape = self.lower_shape(inner)?;
                let total: usize = shape.iter().product();
                let shape_opt = if shape.len() > 1 { Some(shape) } else { None };
                Ok((total, shape_opt))
            }
            Rule::ident => {
                let id = inner.as_str();
                if let Some(&(w, sh)) = self.type_aliases.get(id) {
                    Ok((w, sh))
                } else {
                    Ok((1, None))
                }
            }
            other => Err(format!("Unknown type spec rule: {:?}", other)),
        }
    }

    pub fn lower_port_item(
        &self,
        pair: Pair<'a, Rule>,
    ) -> Result<(&'a str, usize, Option<&'a [usize]>), String> {
        let mut inner = pair.into_inner();
        let name = inner.next().unwrap().as_str();
        let mut width = 1;
        let mut shape = None;

        for p in inner {
            match p.as_rule() {
                Rule::shape => {
                    let sh = self.lower_shape(p)?;
                    width = sh.iter().product();
                    shape = if sh.len() > 1 { Some(sh) } else { None };
                }
                Rule::type_spec => {
                    let (w, s) = self.lower_type_spec(p)?;
                    width = w;
                    shape = s;
                }
                _ => {}
            }
        }
        Ok((name, width, shape))
    }

    pub fn lower_node_dynamics(&self, pair: Pair<'a, Rule>) -> Result<NodeDynamics, String> {
        let inner = pair.into_inner().next().unwrap();
        match inner.as_rule() {
            Rule::dyn_leak => {
                let text = inner.into_inner().next().unwrap().as_str();
                let rate = text.parse::<f64>().unwrap_or(0.0);
                Ok(NodeDynamics::Leak(rate))
            }
            Rule::dyn_latch => Ok(NodeDynamics::Latch),
            Rule::dyn_hold => Ok(NodeDynamics::Hold),
            Rule::dyn_plastic => {
                let mut nums = inner.into_inner();
                let rate = nums
                    .next()
                    .map(|n| n.as_str().parse::<f64>().unwrap())
                    .unwrap_or(0.1);
                let decay = nums
                    .next()
                    .map(|n| n.as_str().parse::<f64>().unwrap())
                    .unwrap_or(0.01);
                Ok(NodeDynamics::Plastic { rate, decay })
            }
            Rule::dyn_osc => {
                let period = inner
                    .into_inner()
                    .next()
                    .unwrap()
                    .as_str()
                    .parse::<usize>()
                    .map_err(|e| e.to_string())?;
                Ok(NodeDynamics::Oscillator {
                    period: period.max(2),
                })
            }
            other => Err(format!("Unknown node dynamics: {:?}", other)),
        }
    }
}
