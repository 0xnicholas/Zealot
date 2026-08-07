//! M5-⑧：pattern（正则）执行——锚定语义的字节级 DFA 编译。
//!
//! 业界惯例（llama.cpp 直接要求 `^…$`；outlines #1083）：pattern 约束整个字符串，
//! 而非 JSON Schema 默认的「搜索」语义。本引擎把 pattern 编译为等价的完整匹配
//! 自动机：`^`/`$` 只允许出现在模式首尾（锚定语义下冗余，接受并忽略）；
//! 不支持的特性（环视/反向引用/命名组/条件/原子组等）显式报错，不静默降级。
//!
//! 支持子集（ECMA-262 常用部分）：
//! - 字面量、`.`（除 `\n`）、字符类 `[...]`/`[^...]`（范围、`]`/`-` 首字面量）
//! - 转义：`\d\D\w\W\s\S`、`\n\t\r\f\v\0`、`\xHH`、`\uHHHH`、元字符 `\\.\\*` 等
//! - 量词 `*` `+` `?` `{n}` `{n,}` `{n,m}`（懒量词 `*?` 等接受——语言等价，
//!   贪婪性对 DFA 无意义；量词叠加 `*+` 等报错）
//! - 分组 `(...)` / `(?:...)`（捕获语义忽略）、交替 `|`
//!
//! 编译管线：解析 → AST → Thompson NFA → 子集构造 DFA → 死状态剪枝。
//! 上限：NFA ≤ MAX_NFA_STATES、DFA ≤ MAX_DFA_STATES，超限显式报错。

/// NFA 状态上限（防爆）。
const MAX_NFA_STATES: usize = 4096;
/// 子集构造后 DFA 状态上限（防指数爆炸）。
const MAX_DFA_STATES: usize = 8192;

// ── AST ─────────────────────────────────────────────────────────────────────

#[derive(Debug)]
enum Ast {
    Empty,
    Char(u8),
    /// 字符类（已展开为闭区间列表；negated 已求反为补集区间）。
    Class(Vec<(u8, u8)>),
    /// `.`：除 `\n` 外任意字节。
    Any,
    Concat(Vec<Ast>),
    Alt(Vec<Ast>),
    Repeat {
        atom: Box<Ast>,
        min: u32,
        max: Option<u32>,
    },
}

// ── 解析 ────────────────────────────────────────────────────────────────────

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

/// 编译 pattern 为字节级 DFA（锚定语义：匹配整个字符串）。
pub fn compile_pattern(pattern: &str) -> Result<RegexDfa, String> {
    // 锚定语义下首尾 ^/$ 冗余：剥离（转义过的 `\^`/`\$` 不剥离）
    let bytes = pattern.as_bytes();
    let mut start = 0;
    let mut end = bytes.len();
    let escaped_at = |i: usize| -> bool {
        let mut n = 0;
        let mut j = i;
        while j > 0 && bytes[j - 1] == b'\\' {
            n += 1;
            j -= 1;
        }
        n % 2 == 1
    };
    if bytes.first() == Some(&b'^') && !escaped_at(0) {
        start = 1;
    }
    if end > start && bytes[end - 1] == b'$' && !escaped_at(end - 1) {
        end -= 1;
    }
    let mut p = Parser {
        bytes: &bytes[start..end],
        pos: 0,
    };
    let ast = p.parse_alt()?;
    if p.pos != p.bytes.len() {
        return Err(format!(
            "pattern 解析失败（位置 {}）：锚点 `^`/`$` 仅允许出现在模式首尾；不支持的语法或未闭合分组/字符类",
            p.pos
        ));
    }
    let nfa = build_nfa(&ast)?;
    subset_construct(&nfa)
}

impl<'a> Parser<'a> {
    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn next(&mut self) -> Option<u8> {
        let b = self.peek();
        if b.is_some() {
            self.pos += 1;
        }
        b
    }

    fn parse_alt(&mut self) -> Result<Ast, String> {
        let mut branches = vec![self.parse_concat()?];
        while self.peek() == Some(b'|') {
            self.pos += 1;
            branches.push(self.parse_concat()?);
        }
        if branches.len() == 1 {
            Ok(branches.pop().unwrap())
        } else {
            Ok(Ast::Alt(branches))
        }
    }

    fn parse_concat(&mut self) -> Result<Ast, String> {
        let mut items = Vec::new();
        loop {
            match self.peek() {
                None | Some(b'|') | Some(b')') => break,
                _ => items.push(self.parse_repeat()?),
            }
        }
        Ok(match items.len() {
            0 => Ast::Empty,
            1 => items.pop().unwrap(),
            _ => Ast::Concat(items),
        })
    }

    fn parse_repeat(&mut self) -> Result<Ast, String> {
        let atom = self.parse_atom()?;
        let (min, max) = match self.peek() {
            Some(b'*') => {
                self.pos += 1;
                (0, None)
            }
            Some(b'+') => {
                self.pos += 1;
                (1, None)
            }
            Some(b'?') => {
                self.pos += 1;
                (0, Some(1))
            }
            Some(b'{') => match self.try_parse_braces() {
                Ok(Some(q)) => q,
                Ok(None) => return Ok(atom), // `{` 不是有效量词 → 字面量
                Err(e) if e.is_empty() => return Ok(atom), // 同上（pos 已回滚）
                Err(e) => return Err(e),
            },
            _ => return Ok(atom),
        };
        // 懒量词（`*?` 等）：语言等价，接受并忽略；量词叠加（`*+` 等）报错
        if self.peek() == Some(b'+') {
            return Err("pattern 不支持的量词叠加（如 `*+`）".into());
        }
        if self.peek() == Some(b'?') {
            self.pos += 1;
        }
        Ok(Ast::Repeat {
            atom: Box::new(atom),
            min,
            max,
        })
    }

    /// 尝试解析 `{n}` / `{n,}` / `{n,m}`；失败（非有效量词形态）返回 None（`{` 按字面量）。
    fn try_parse_braces(&mut self) -> Result<Option<(u32, Option<u32>)>, String> {
        let save = self.pos;
        self.pos += 1; // consume '{'
        let min = self.parse_digits().ok_or_else(|| {
            self.pos = save;
            String::new()
        })?;
        let max = match self.peek() {
            Some(b'}') => {
                self.pos += 1;
                Some(min)
            }
            Some(b',') => {
                self.pos += 1;
                match self.peek() {
                    Some(b'}') => {
                        self.pos += 1;
                        None
                    }
                    _ => {
                        let m = self.parse_digits().ok_or_else(|| {
                            self.pos = save;
                            String::new()
                        })?;
                        if self.next() != Some(b'}') {
                            self.pos = save;
                            return Err(String::new());
                        }
                        Some(m)
                    }
                }
            }
            _ => {
                self.pos = save;
                return Err(String::new());
            }
        };
        if max.is_some_and(|m| m < min) {
            return Err(format!(
                "pattern 无效量词 {{{min},{}}}：下限大于上限",
                max.unwrap()
            ));
        }
        Ok(Some((min, max)))
    }

    fn parse_digits(&mut self) -> Option<u32> {
        let start = self.pos;
        let mut v: u32 = 0;
        while let Some(b @ b'0'..=b'9') = self.peek() {
            v = v.checked_mul(10)?.checked_add((b - b'0') as u32)?;
            self.pos += 1;
        }
        if self.pos == start {
            None
        } else {
            Some(v)
        }
    }

    fn parse_atom(&mut self) -> Result<Ast, String> {
        let b = self.next().ok_or("pattern 空模式")?;
        match b {
            b'(' => {
                if self.peek() == Some(b'?') {
                    self.pos += 1;
                    match self.next() {
                        Some(b':') => {} // 非捕获组
                        Some(b'=') | Some(b'!') | Some(b'<') => {
                            return Err("pattern 不支持环视（lookahead/lookbehind）与命名组".into());
                        }
                        Some(b'>') => return Err("pattern 不支持原子组".into()),
                        Some(c) => {
                            return Err(format!("pattern 不支持的组语法 `(?{}`", c as char));
                        }
                        None => return Err("pattern 未闭合的分组".into()),
                    }
                }
                let inner = self.parse_alt()?;
                if self.next() != Some(b')') {
                    return Err("pattern 未闭合的分组".into());
                }
                Ok(inner)
            }
            b')' => Err("pattern 多余的 `)`".into()),
            b'[' => self.parse_class(),
            b'.' => Ok(Ast::Any),
            b'^' | b'$' => {
                Err("pattern 锚点 `^`/`$` 仅允许出现在模式首尾（锚定语义下冗余）".into())
            }
            b'*' | b'+' | b'?' => Err("pattern 量词缺少原子（如 `a*` 中的 `a`）".into()),
            b'\\' => self.parse_escape(),
            b'{' => {
                // `{` 后跟有效量词形态 → 量词缺原子；否则字面量
                let mut probe = Parser {
                    bytes: self.bytes,
                    pos: self.pos + 1,
                };
                match probe.try_parse_braces() {
                    Ok(Some(_)) => Err("pattern 量词缺少原子（如 `a{2}` 中的 `a`）".into()),
                    _ => Ok(Ast::Char(b'{')),
                }
            }
            c => Ok(Ast::Char(c)),
        }
    }

    fn parse_class(&mut self) -> Result<Ast, String> {
        let negated = self.peek() == Some(b'^');
        if negated {
            self.pos += 1;
        }
        let mut ranges: Vec<(u8, u8)> = Vec::new();
        // 首位置的 `]`/`-` 为字面量
        if self.peek() == Some(b']') {
            self.pos += 1;
            ranges.push((b']', b']'));
        } else if self.peek() == Some(b'-') {
            self.pos += 1;
            ranges.push((b'-', b'-'));
        }
        loop {
            let b = self.next().ok_or("pattern 未闭合的字符类 `[...]`")?;
            if b == b']' {
                break;
            }
            let lo = if b == b'\\' {
                self.parse_class_escape()?
            } else {
                b
            };
            if self.peek() == Some(b'-') && self.bytes.get(self.pos + 1) != Some(&b']') {
                self.pos += 1; // consume '-'
                let hi = self.next().ok_or("pattern 未闭合的字符类")?;
                let hi = if hi == b'\\' {
                    self.parse_class_escape()?
                } else {
                    hi
                };
                if hi < lo {
                    return Err("pattern 无效字符类范围（下限大于上限）".into());
                }
                ranges.push((lo, hi));
            } else {
                ranges.push((lo, lo));
            }
        }
        if ranges.is_empty() {
            // 空类（如 `[]`）永远不匹配——用空区间表示
            return Ok(Ast::Class(vec![]));
        }
        if negated {
            ranges = complement_ranges(&ranges);
        }
        Ok(Ast::Class(ranges))
    }

    fn parse_class_escape(&mut self) -> Result<u8, String> {
        let b = self.next().ok_or("pattern 转义缺少字符")?;
        match b {
            b'n' => Ok(b'\n'),
            b't' => Ok(b'\t'),
            b'r' => Ok(b'\r'),
            b'f' => Ok(b'\x0c'),
            b'v' => Ok(b'\x0b'),
            b'0' => Ok(0),
            b'b' => Ok(0x08), // ECMA：类内 \b = 退格
            b'x' => self.parse_hex(2, "\\x"),
            b'u' => self.parse_hex(4, "\\u"),
            c => Ok(c), // 类内 \d 等按字面量（ECMA 语义），其余转义即字面
        }
    }

    fn parse_escape(&mut self) -> Result<Ast, String> {
        let b = self.next().ok_or("pattern 转义缺少字符")?;
        match b {
            b'd' => Ok(Ast::Class(vec![(b'0', b'9')])),
            b'D' => Ok(Ast::Class(complement_ranges(&[(b'0', b'9')]))),
            b'w' => Ok(Ast::Class(vec![
                (b'0', b'9'),
                (b'A', b'Z'),
                (b'_', b'_'),
                (b'a', b'z'),
            ])),
            b'W' => Ok(Ast::Class(complement_ranges(&[
                (b'0', b'9'),
                (b'A', b'Z'),
                (b'_', b'_'),
                (b'a', b'z'),
            ]))),
            b's' => Ok(Ast::Class(vec![
                (b' ', b' '),
                (b'\t', b'\t'),
                (b'\n', b'\n'),
                (b'\r', b'\r'),
                (b'\x0c', b'\x0c'),
                (b'\x0b', b'\x0b'),
            ])),
            b'S' => Ok(Ast::Class(complement_ranges(&[
                (b' ', b' '),
                (b'\t', b'\t'),
                (b'\n', b'\n'),
                (b'\r', b'\r'),
                (b'\x0c', b'\x0c'),
                (b'\x0b', b'\x0b'),
            ]))),
            b'n' => Ok(Ast::Char(b'\n')),
            b't' => Ok(Ast::Char(b'\t')),
            b'r' => Ok(Ast::Char(b'\r')),
            b'f' => Ok(Ast::Char(b'\x0c')),
            b'v' => Ok(Ast::Char(b'\x0b')),
            b'0' => Ok(Ast::Char(0)),
            b'x' => Ok(Ast::Char(self.parse_hex(2, "\\x")?)),
            b'u' => Ok(Ast::Char(self.parse_hex(4, "\\u")?)),
            b'1'..=b'9' => Err("pattern 不支持反向引用（`\\1` 等）".into()),
            b'b' | b'B' | b'A' | b'z' | b'Z' | b'G' => {
                Err("pattern 不支持词边界/串锚点转义（`\\b` 等）——锚定语义下用 `^`/`$`".into())
            }
            c => Ok(Ast::Char(c)), // 元字符转义即字面
        }
    }

    fn parse_hex(&mut self, len: usize, what: &str) -> Result<u8, String> {
        let mut v: u32 = 0;
        for _ in 0..len {
            let b = self.next().ok_or(format!("pattern {what} 转义位数不足"))?;
            let d = match b {
                b'0'..=b'9' => b - b'0',
                b'a'..=b'f' => b - b'a' + 10,
                b'A'..=b'F' => b - b'A' + 10,
                _ => return Err(format!("pattern {what} 转义含非十六进制字符")),
            };
            v = v * 16 + d as u32;
        }
        Ok(v as u8)
    }
}

/// 区间补集（0..=255 减去给定闭区间）。
fn complement_ranges(ranges: &[(u8, u8)]) -> Vec<(u8, u8)> {
    let mut out = Vec::new();
    let mut cur: u16 = 0;
    for &(lo, hi) in ranges {
        let lo16 = lo as u16;
        let hi16 = hi as u16;
        if cur < lo16 {
            out.push((cur as u8, (lo16 - 1) as u8));
        }
        cur = cur.max(hi16 + 1);
    }
    if cur <= 255 {
        out.push((cur as u8, 255));
    }
    out
}

// ── NFA（Thompson）──────────────────────────────────────────────────────────

struct Nfa {
    /// transitions[s] = 字节边 (byte → target)。
    byte_edges: Vec<Vec<(u8, usize)>>,
    /// transitions[s] = ε 边。
    eps: Vec<Vec<usize>>,
    accept: Vec<bool>,
}

impl Nfa {
    fn new_state(&mut self) -> usize {
        let id = self.byte_edges.len();
        self.byte_edges.push(Vec::new());
        self.eps.push(Vec::new());
        self.accept.push(false);
        id
    }

    fn add_byte(&mut self, from: usize, b: u8, to: usize) {
        self.byte_edges[from].push((b, to));
    }

    fn add_eps(&mut self, from: usize, to: usize) {
        self.eps[from].push(to);
    }
}

/// 构建 NFA（start = 状态 0，accept 标记唯一收尾状态）。
fn build_nfa(ast: &Ast) -> Result<Nfa, String> {
    let mut nfa = Nfa {
        byte_edges: Vec::new(),
        eps: Vec::new(),
        accept: Vec::new(),
    };
    let start = nfa.new_state();
    let end = nfa.new_state();
    nfa.accept[end] = true;
    emit(&mut nfa, ast, start, end)?;
    if nfa.byte_edges.len() > MAX_NFA_STATES {
        return Err(format!("pattern 过大（NFA 超过 {} 状态）", MAX_NFA_STATES));
    }
    Ok(nfa)
}

/// 把 AST 展开到 [from, to] 之间的 NFA 片段。
fn emit(nfa: &mut Nfa, ast: &Ast, from: usize, to: usize) -> Result<(), String> {
    match ast {
        Ast::Empty => nfa.add_eps(from, to),
        Ast::Char(c) => nfa.add_byte(from, *c, to),
        Ast::Any => {
            for b in 0..=255u8 {
                if b != b'\n' {
                    nfa.add_byte(from, b, to);
                }
            }
        }
        Ast::Class(ranges) => {
            for &(lo, hi) in ranges {
                for b in lo..=hi {
                    nfa.add_byte(from, b, to);
                }
            }
        }
        Ast::Concat(items) => {
            let mut cur = from;
            for (i, item) in items.iter().enumerate() {
                let next = if i == items.len() - 1 {
                    to
                } else {
                    nfa.new_state()
                };
                emit(nfa, item, cur, next)?;
                cur = next;
            }
        }
        Ast::Alt(branches) => {
            for branch in branches {
                let (s, e) = (nfa.new_state(), nfa.new_state());
                nfa.add_eps(from, s);
                emit(nfa, branch, s, e)?;
                nfa.add_eps(e, to);
            }
        }
        Ast::Repeat { atom, min, max } => {
            let min = *min as usize;
            let max = *max;
            // 前置 min 个必需副本
            let mut cur = from;
            for _ in 0..min {
                let next = nfa.new_state();
                emit(nfa, atom, cur, next)?;
                cur = next;
            }
            match max {
                Some(m) => {
                    let m = m as usize;
                    // 附加 (m - min) 个顺序可选副本（每个：消费一次或跳过）
                    for _ in 0..(m.saturating_sub(min)) {
                        let (s, e, next) = (nfa.new_state(), nfa.new_state(), nfa.new_state());
                        nfa.add_eps(cur, s);
                        emit(nfa, atom, s, e)?;
                        nfa.add_eps(e, next);
                        nfa.add_eps(cur, next); // 跳过本次
                        cur = next;
                    }
                    nfa.add_eps(cur, to);
                }
                None => {
                    // 开放循环：cur →(atom)→ e →(ε)→ cur
                    let (s, e) = (nfa.new_state(), nfa.new_state());
                    nfa.add_eps(cur, s);
                    emit(nfa, atom, s, e)?;
                    nfa.add_eps(e, s);
                    nfa.add_eps(e, to);
                    nfa.add_eps(cur, to); // 0 次
                }
            }
        }
    }
    Ok(())
}

// ── 子集构造 → DFA ──────────────────────────────────────────────────────────

/// 编译产物：抽象 DFA（字节边 + 终态）。
pub struct RegexDfa {
    pub start: usize,
    pub finals: Vec<usize>,
    /// transitions[s] = 排序后的 (byte, target)。
    pub transitions: Vec<Vec<(u8, usize)>>,
}

fn subset_construct(nfa: &Nfa) -> Result<RegexDfa, String> {
    use std::collections::{BTreeSet, HashMap, VecDeque};

    // ε-闭包
    fn closure(nfa: &Nfa, set: &BTreeSet<usize>) -> BTreeSet<usize> {
        let mut out = set.clone();
        let mut stack: Vec<usize> = out.iter().copied().collect();
        while let Some(s) = stack.pop() {
            for &t in &nfa.eps[s] {
                if out.insert(t) {
                    stack.push(t);
                }
            }
        }
        out
    }

    let mut ids: HashMap<BTreeSet<usize>, usize> = HashMap::new();
    let mut sets: Vec<BTreeSet<usize>> = Vec::new();
    let mut trans: Vec<Vec<(u8, usize)>> = Vec::new();
    let mut finals: Vec<bool> = Vec::new();
    let mut queue: VecDeque<usize> = VecDeque::new();

    let start_set = closure(nfa, &[0usize].into_iter().collect());
    ids.insert(start_set.clone(), 0);
    sets.push(start_set.clone());
    trans.push(Vec::new());
    finals.push(start_set.iter().any(|&s| nfa.accept[s]));
    queue.push_back(0);

    while let Some(id) = queue.pop_front() {
        let set: BTreeSet<usize> = sets[id].clone();
        // 按字节计算 move
        for b in 0..=255u8 {
            let mut next: BTreeSet<usize> = BTreeSet::new();
            for &s in &set {
                for &(bb, t) in &nfa.byte_edges[s] {
                    if bb == b {
                        next.insert(t);
                    }
                }
            }
            if next.is_empty() {
                continue;
            }
            let next = closure(nfa, &next);
            let next_id = if let Some(&nid) = ids.get(&next) {
                nid
            } else {
                let nid = trans.len();
                if nid >= MAX_DFA_STATES {
                    return Err(format!("pattern 过大（DFA 超过 {} 状态）", MAX_DFA_STATES));
                }
                ids.insert(next.clone(), nid);
                sets.push(next.clone());
                trans.push(Vec::new());
                finals.push(next.iter().any(|&s| nfa.accept[s]));
                queue.push_back(nid);
                nid
            };
            trans[id].push((b, next_id));
        }
        trans[id].sort_unstable_by_key(|&(b, _)| b);
    }

    // 死状态剪枝：无出边且非终态的子集不可达成功——去掉指向它们的边
    let n = trans.len();
    let mut alive = vec![false; n];
    for i in 0..n {
        if finals[i] {
            alive[i] = true;
        }
    }
    // 反向迭代：只有出边全部指向 alive 且自身…——简化：迭代直到不动点
    loop {
        let mut changed = false;
        for i in 0..n {
            if alive[i] {
                continue;
            }
            if trans[i].iter().any(|&(_, t)| alive[t]) {
                alive[i] = true;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    for edges in trans.iter_mut() {
        edges.retain(|&(_, t)| alive[t]);
    }
    // 重编号：去除死状态
    let mut remap = vec![usize::MAX; n];
    let mut next_id = 0;
    for i in 0..n {
        if alive[i] {
            remap[i] = next_id;
            next_id += 1;
        }
    }
    let mut new_trans: Vec<Vec<(u8, usize)>> = vec![Vec::new(); next_id];
    for i in 0..n {
        if !alive[i] {
            continue;
        }
        let ni = remap[i];
        for &(b, t) in &trans[i] {
            new_trans[ni].push((b, remap[t]));
        }
    }
    let start = remap[0];
    let finals = (0..n)
        .filter(|&i| alive[i] && finals[i])
        .map(|i| remap[i])
        .collect();
    Ok(RegexDfa {
        start,
        finals,
        transitions: new_trans,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn accepts(pat: &str, input: &str) -> bool {
        let dfa = compile_pattern(pat).expect("compile");
        let mut st = dfa.start;
        for b in input.as_bytes() {
            let t = dfa.transitions[st]
                .binary_search_by_key(b, |&(bb, _)| bb)
                .ok()
                .map(|i| dfa.transitions[st][i].1);
            match t {
                Some(n) => st = n,
                None => return false,
            }
        }
        dfa.finals.contains(&st)
    }

    fn rejects(pat: &str, input: &str) -> bool {
        !accepts(pat, input)
    }

    #[test]
    fn test_literals_anchored() {
        assert!(accepts("abc", "abc"));
        assert!(rejects("abc", "xabcx"), "锚定语义：整体匹配");
        assert!(rejects("abc", "ab"));
        assert!(accepts("^abc$", "abc"));
        assert!(rejects("^abc", "xabc"));
    }

    #[test]
    fn test_classes_and_dot() {
        assert!(accepts("[a-z]+", "abc"));
        assert!(rejects("[a-z]+", "a1"));
        assert!(accepts("[^0-9]+", "abc"));
        assert!(rejects("[^0-9]+", "a1b"));
        assert!(accepts("[a-cx-z]{2}", "bx"));
        assert!(rejects("[a-cx-z]{2}", "bd"));
        assert!(accepts(".+", "ab"));
        assert!(rejects(".", "\n"));
        assert!(accepts("[]]", "]"), "首位置 ] 为字面量");
        assert!(accepts("[-a]", "-"));
        assert!(accepts("a{", "a{"), "无效量词花括号按字面量");
    }

    #[test]
    fn test_quantifiers() {
        assert!(accepts("a*", ""));
        assert!(accepts("a*", "aaa"));
        assert!(accepts("a+", "a"));
        assert!(rejects("a+", ""));
        assert!(accepts("a?", ""));
        assert!(accepts("a?b", "ab"));
        assert!(accepts("a{2,3}", "aa"));
        assert!(accepts("a{2,3}", "aaa"));
        assert!(rejects("a{2,3}", "a"));
        assert!(rejects("a{2,3}", "aaaa"));
        assert!(accepts("a{2}", "aa"));
        assert!(accepts("a{2,}", "aaaa"));
        assert!(rejects("a{2,}", "a"));
        assert!(accepts("(ab)+", "abab"));
        assert!(rejects("(ab)+", "ababa"));
    }

    #[test]
    fn test_alternation_groups() {
        assert!(accepts("a|b", "a"));
        assert!(accepts("a|b", "b"));
        assert!(rejects("a|b", "c"));
        assert!(accepts("(a|bc)d", "bcd"));
        assert!(accepts("(?:ab)+", "abab"));
        assert!(accepts("a|", ""));
        assert!(accepts("(|a)", ""));
    }

    #[test]
    fn test_escapes() {
        assert!(accepts(r"\d{3}-\d{2}", "123-45"));
        assert!(rejects(r"\d{3}-\d{2}", "12-345"));
        assert!(accepts(r"\w+", "abc_123"));
        assert!(rejects(r"\w+", "a b"));
        assert!(accepts(r"\s+", " \t\n"));
        assert!(accepts(r"\u0041", "A"));
        assert!(accepts(r"\x41", "A"));
        assert!(accepts(r"a\.b", "a.b"));
        assert!(rejects(r"a\.b", "axb"));
        assert!(accepts(r"\D+", "abc"));
        assert!(rejects(r"\D+", "a1"));
    }

    #[test]
    fn test_unsupported_features_error() {
        for pat in [
            r"(?=a)",
            r"(?!a)",
            r"(?<=a)",
            r"(?<name>a)",
            r"(?>a)",
            r"a\1",
            r"a\b",
            r"a|^b",
            r"a$|b",
            "(", // 未闭合
            "[", // 未闭合
            ")", // 多余
            r"a{3,1}",
            r"a*+",
            r"**",
        ] {
            let err = compile_pattern(pat);
            assert!(err.is_err(), "pattern {pat:?} 应报错");
        }
    }

    #[test]
    fn test_utf8_bytes() {
        // 多字节字符按字节匹配（字节级自动机）
        assert!(accepts("中", "中"));
        assert!(accepts(".+", "中文"));
    }
}
