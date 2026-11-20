//! PS9.4 experimental nonlinear causal predictor synthesis.
//! This module is standalone from the archive selector until its search cost
//! and serialized descriptor economics are measured in the full encoder.



pub const TERM_LINEAR_1: u8 = 0;
pub const TERM_LINEAR_2: u8 = 1;
pub const TERM_SQUARE_1: u8 = 2;
pub const TERM_PRODUCT_12: u8 = 3;
pub const TERM_XOR_12: u8 = 4;
pub const TERM_AND_12: u8 = 5;
pub const TERM_OR_12: u8 = 6;
pub const TERM_CUBE_1: u8 = 7;
pub const MAX_TERMS: usize = 8;


#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AlgebraicExpression { pub terms: Vec<(u8, u8)> }


fn term_value(term: u8, history: &[u8]) -> u8 {
    let p1 = history.last().copied().unwrap_or(0);
    let p2 = if history.len() >= 2 { history[history.len()-2] } else { 0 };
    match term {
        TERM_LINEAR_1 => p1,
        TERM_LINEAR_2 => p2,
        TERM_SQUARE_1 => p1.wrapping_mul(p1),
        TERM_PRODUCT_12 => p1.wrapping_mul(p2),
        TERM_XOR_12 => p1 ^ p2,
        TERM_AND_12 => p1 & p2,
        TERM_OR_12 => p1 | p2,
        TERM_CUBE_1 => p1.wrapping_mul(p1).wrapping_mul(p1),
        _ => 0,
    }
}
impl AlgebraicExpression {
    pub fn validate(&self) -> bool {
        self.terms.len() <= MAX_TERMS && self.terms.iter().all(|(t,_)| *t <= TERM_CUBE_1)
            && self.terms.windows(2).all(|w| w[0].0 < w[1].0)
    }
    pub fn encode_parameters(&self) -> Option<Vec<u8>> {
        if !self.validate() { return None; }
        let mut out = Vec::with_capacity(1 + self.terms.len()*2);
        out.push(self.terms.len() as u8);
        for &(term, coefficient) in &self.terms { out.push(term); out.push(coefficient); }
        Some(out)
    }
    pub fn decode_parameters(p: &[u8]) -> Option<Self> {
        if p.is_empty() || p[0] as usize > MAX_TERMS || p.len() != 1 + p[0] as usize * 2 { return None; }
        let terms = (0..p[0] as usize).map(|i| (p[1+i*2], p[2+i*2])).collect();
        let model = Self { terms };
        if model.validate() { Some(model) } else { None }
    }
    pub fn predict(&self, history: &[u8]) -> u8 {
        self.terms.iter().fold(0u8, |sum, &(t,c)| sum.wrapping_add(term_value(t,history).wrapping_mul(c)))
    }
    pub fn forward(&self, input: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(input.len());
        for i in 0..input.len() { out.push(input[i].wrapping_sub(self.predict(&input[..i]))); }
        out
    }
    pub fn inverse(&self, residual: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(residual.len());
        for &r in residual { let p = self.predict(&out); out.push(r.wrapping_add(p)); }
        out
    }
}

/// Deterministic multi-start coordinate search on a Renyi-2 histogram score.
/// Final acceptance must be performed by the full archive-cost evaluator.
pub fn synthesize(sample: &[u8], max_terms: usize, rounds: usize) -> AlgebraicExpression {
    let data = &sample[..sample.len().min(2048)];
    let max_terms = max_terms.min(MAX_TERMS);
    let score = |m: &AlgebraicExpression| -> u128 {
        let residual = m.forward(data);
        let mut h = [0u32; 256];
        for b in residual { h[b as usize] += 1; }
        h.iter().map(|&v| (v as u128) * (v as u128)).sum()
    };
    let mut best = AlgebraicExpression { terms: Vec::new() };
    let mut best_score = score(&best);
    // Start from the PS9.3 linear recurrence as a family of seeded candidates.
    for seed in 0..4 {
        let mut model = AlgebraicExpression { terms: vec![(TERM_LINEAR_1, (seed*53) as u8), (TERM_LINEAR_2, (seed*97) as u8)] };
        model.terms.retain(|&(_,c)| c != 0);
        let mut current = score(&model);
        for _ in 0..rounds.max(1) {
            for term in 0..=TERM_CUBE_1 {
                if model.terms.len() >= max_terms && !model.terms.iter().any(|&(t,_)| t==term) { continue; }
                let old = model.terms.iter().position(|&(t,_)| t==term).map(|i| model.terms[i].1);
                let mut chosen = old.unwrap_or(0); let mut chosen_score = current;
                for c in 0u16..=255 {
                    let mut trial = model.clone();
                    if let Some(pos) = trial.terms.iter().position(|&(t,_)| t==term) { trial.terms[pos].1=c as u8; }
                    else if c != 0 { trial.terms.push((term,c as u8)); trial.terms.sort_by_key(|x|x.0); }
                    trial.terms.retain(|&(_,coef)| coef != 0);
                    let s=score(&trial);
                    if s > chosen_score { chosen=c as u8; chosen_score=s; }
                }
                if chosen != old.unwrap_or(0) {
                    if let Some(pos)=model.terms.iter().position(|&(t,_)|t==term) { model.terms[pos].1=chosen; }
                    else if chosen != 0 { model.terms.push((term,chosen)); model.terms.sort_by_key(|x|x.0); }
                    model.terms.retain(|&(_,c)|c!=0); current=chosen_score;
                }
            }
        }
        if current > best_score { best=model; best_score=current; }
    }
    best
}

/// Diagnostic histogram entropy proxy (bits, floating point); not used for decisions.
pub fn histogram_entropy(data: &[u8]) -> f64 {
    let mut counts = [0usize; 256];
    for &b in data { counts[b as usize] += 1; }
    let n=data.len() as f64;
    if n==0.0 { return 0.0; }
    counts.iter().filter(|&&c|c>0).map(|&c| { let p=c as f64/n; -p*p.log2() }).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn nonlinear_expression_roundtrips_every_byte() {
        let m=AlgebraicExpression { terms: vec![(TERM_LINEAR_1,7),(TERM_SQUARE_1,11),(TERM_PRODUCT_12,19),(TERM_XOR_12,3)] };
        let x: Vec<u8>=(0..8192).map(|i| ((i*31 + (i/9)*17) & 255) as u8).collect();
        assert_eq!(m.inverse(&m.forward(&x)),x);
    }
    #[test]
    fn parameters_are_canonical_and_validated() {
        let m=AlgebraicExpression { terms: vec![(TERM_LINEAR_1,1),(TERM_CUBE_1,5)] };
        assert_eq!(AlgebraicExpression::decode_parameters(&m.encode_parameters().unwrap()),Some(m));
        assert!(AlgebraicExpression::decode_parameters(&[2,3,4,1,2]).is_none());
        assert!(AlgebraicExpression { terms: vec![(2,1),(1,2)] }.encode_parameters().is_none());
    }
    #[test]
    fn predictor_synthesis_is_deterministic_and_roundtrips() {
        let x: Vec<u8>=(0..2048).map(|i| ((i*13 + (i/5)*7)&255) as u8).collect();
        let a=synthesize(&x,5,1); let b=synthesize(&x,5,1);
        assert_eq!(a,b); assert_eq!(a.inverse(&a.forward(&x)),x);
    }
}
