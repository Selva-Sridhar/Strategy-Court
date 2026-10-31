"""Static scan of strategy source code for look-ahead patterns and unsafe constructs.

Static analysis raises *suspicion*; only the dynamic causality test (causality.py) proves
look-ahead. Findings are therefore capped at HIGH severity and every look-ahead finding says it
is verified dynamically. Each rule documents what it deliberately does not flag, to keep false
positives on legitimate code (rolling windows, cross-sectional operations) at zero.

Rules
-----
lookahead.negative_shift   .shift/.diff/.pct_change/np.roll by a negative amount (constants
                           are resolved through simple assignments and defaults)
lookahead.centered_window  .rolling(..., center=True)
lookahead.backfill         .bfill() / .backfill() / fillna or reindex with method="bfill",
                           .interpolate() (uses later observations)
lookahead.reverse_time     slicing with a negative step, e.g. [::-1]
lookahead.full_sample_stat time-series aggregation over the whole sample (.mean(), .std(),
                           .quantile(), .rank(), ...) not inside rolling/expanding/ewm/groupby
                           and not cross-sectional (axis=1)
lookahead.full_sample_fit  .fit / .fit_transform / .fit_predict on data that is not sliced
lookahead.end_of_sample    .iloc[-k] (the last rows are the future for every earlier bar)
execution.same_bar_pnl     positions multiplied by same-bar returns without a positive shift
determinism.unseeded       np.random legacy calls / default_rng() / random.* without a seed
hindsight.hardcoded_date   date literals (strategies keyed to known historical events)
security.forbidden_import  imports outside the sandbox allowlist (network, OS, data fetching)
security.dangerous_call    open/eval/exec/compile/__import__/globals/... and dunder escapes
"""

import ast
import re
import textwrap
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from strategy_court.audit._common import make_evidence
from strategy_court.audit.citations import ARNOTT_HARVEY_MARKOWITZ_2019, KAUFMAN_2012
from strategy_court.schemas import Evidence, Severity

TOOL = "ast_scanner"
CITATIONS = (KAUFMAN_2012, ARNOTT_HARVEY_MARKOWITZ_2019)
Category = Literal["lookahead", "execution", "determinism", "hindsight", "security"]

ALLOWED_IMPORTS = frozenset(
    {
        "numpy",
        "pandas",
        "math",
        "statistics",
        "typing",
        "dataclasses",
        "collections",
        "itertools",
        "functools",
        "enum",
        "abc",
        "datetime",
        "operator",
        "__future__",
    }
)
DATA_FETCH = frozenset({"yfinance", "requests", "urllib", "httpx", "aiohttp", "pandas_datareader"})
DANGEROUS_CALLS = frozenset(
    {
        "open",
        "eval",
        "exec",
        "compile",
        "__import__",
        "globals",
        "locals",
        "vars",
        "input",
        "breakpoint",
        "setattr",
        "delattr",
    }
)
DUNDER_ESCAPES = frozenset(
    {
        "__class__",
        "__subclasses__",
        "__globals__",
        "__builtins__",
        "__bases__",
        "__mro__",
        "__code__",
        "__getattribute__",
        "__dict__",
    }
)
SHIFT_LIKE = frozenset({"shift", "diff", "pct_change"})
AGGREGATIONS = frozenset(
    {
        "mean",
        "std",
        "var",
        "median",
        "quantile",
        "min",
        "max",
        "sum",
        "rank",
        "describe",
        "idxmax",
        "idxmin",
        "nlargest",
        "nsmallest",
        "skew",
        "kurt",
    }
)
WINDOWED = frozenset({"rolling", "expanding", "ewm", "groupby", "resample"})
FIT = frozenset({"fit", "fit_transform", "fit_predict"})
LEGACY_RANDOM = frozenset(
    {
        "rand",
        "randn",
        "random",
        "normal",
        "uniform",
        "choice",
        "permutation",
        "shuffle",
        "randint",
        "standard_normal",
        "binomial",
        "poisson",
        "exponential",
        "sample",
    }
)
DATE_RE = re.compile(r"^(19[9]\d|20[0-3]\d)[-/](0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])")


@dataclass(frozen=True)
class Finding:
    rule: str
    category: Category
    severity: Severity
    line: int
    col: int
    snippet: str
    message: str


def _const_env(tree: ast.AST) -> dict[str, float]:
    """Names bound exactly once to a numeric constant expression (incl. default arguments)."""
    seen: dict[str, list[ast.expr]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            t = node.targets[0]
            if isinstance(t, ast.Name):
                seen.setdefault(t.id, []).append(node.value)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value:
            seen.setdefault(node.target.id, []).append(node.value)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            args = node.args.args[-len(node.args.defaults) :] if node.args.defaults else []
            for a, d in zip(args, node.args.defaults, strict=True):
                seen.setdefault(a.arg, []).append(d)
            for a, kd in zip(node.args.kwonlyargs, node.args.kw_defaults, strict=True):
                if kd is not None:
                    seen.setdefault(a.arg, []).append(kd)
    env: dict[str, float] = {}
    for _ in range(3):  # allow constants defined from other constants
        for name, values in seen.items():
            evaluated = {_eval(v, env) for v in values}
            if len(evaluated) == 1 and None not in evaluated:
                val = evaluated.pop()
                assert val is not None
                env[name] = val
    return env


def _eval(node: ast.expr, env: dict[str, float]) -> float | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return None if isinstance(node.value, bool) else float(node.value)
    if isinstance(node, ast.Name):
        return env.get(node.id)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub | ast.UAdd):
        v = _eval(node.operand, env)
        return None if v is None else (-v if isinstance(node.op, ast.USub) else v)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add | ast.Sub | ast.Mult):
        a, b = _eval(node.left, env), _eval(node.right, env)
        if a is None or b is None:
            return None
        return (
            a + b
            if isinstance(node.op, ast.Add)
            else a - b
            if isinstance(node.op, ast.Sub)
            else a * b
        )
    return None


def _call_name(node: ast.Call) -> str | None:
    f = node.func
    return f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else None


def _dotted(node: ast.expr) -> str:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _chain_calls(node: ast.expr) -> list[str]:
    """Method names called along a receiver chain, e.g. x.rolling(5).mean() -> [mean, rolling]."""
    out = []
    while True:
        if isinstance(node, ast.Call):
            name = _call_name(node)
            if name:
                out.append(name)
            node = node.func
        elif isinstance(node, ast.Attribute | ast.Subscript):
            node = node.value
        else:
            return out


def _kw(node: ast.Call, name: str) -> ast.expr | None:
    return next((k.value for k in node.keywords if k.arg == name), None)


def _is_true(node: ast.expr | None) -> bool:
    return isinstance(node, ast.Constant) and node.value is True


def _str_const(node: ast.expr | None) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


class _Scanner(ast.NodeVisitor):
    def __init__(self, source: str, tree: ast.AST) -> None:
        self.lines = source.splitlines()
        self.env = _const_env(tree)
        self.findings: list[Finding] = []
        self.return_names: set[str] = set()
        self.row_names: set[str] = set()  # names bound to a single row (iloc[k]): cross-sectional
        self.seeded_np = self.seeded_random = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and self._selects_single_row(node.value):
                self.row_names.update(t.id for t in node.targets if isinstance(t, ast.Name))
            if isinstance(node, ast.Call):
                dotted = _dotted(node.func)
                if dotted.endswith("random.seed"):
                    if dotted.startswith(("np.", "numpy.")):
                        self.seeded_np = True
                    else:
                        self.seeded_random = True
            if (
                isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Call)
                and (
                    set(_chain_calls(node.value)) & {"pct_change", "diff"}
                    or "return" in _dotted(node.value.func)
                )
            ):
                self.return_names.update(t.id for t in node.targets if isinstance(t, ast.Name))

    def _selects_single_row(self, node: ast.expr) -> bool:
        for sub in ast.walk(node):
            if (
                isinstance(sub, ast.Subscript)
                and isinstance(sub.value, ast.Attribute)
                and sub.value.attr == "iloc"
                and not isinstance(sub.slice, ast.Slice | ast.Tuple)
                and _eval(sub.slice, self.env) is not None
            ):
                return True
        return False

    def add(self, rule: str, cat: Category, sev: Severity, node: ast.AST, msg: str) -> None:
        line = getattr(node, "lineno", 0)
        snippet = self.lines[line - 1].strip()[:160] if 0 < line <= len(self.lines) else ""
        self.findings.append(
            Finding(rule, cat, sev, line, getattr(node, "col_offset", 0), snippet, msg)
        )

    # ---------------------------------------------------------------- imports & calls

    def _check_module(self, module: str, node: ast.AST) -> None:
        root = module.split(".")[0]
        if root in DATA_FETCH:
            self.add(
                "security.forbidden_import",
                "security",
                Severity.HIGH,
                node,
                f"imports '{module}': fetches external data the court cannot verify",
            )
        elif root not in ALLOWED_IMPORTS:
            self.add(
                "security.forbidden_import",
                "security",
                Severity.HIGH,
                node,
                f"imports '{module}', outside the sandbox allowlist",
            )

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._check_module(alias.name, node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self._check_module(node.module or "", node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in DUNDER_ESCAPES:
            self.add(
                "security.dangerous_call",
                "security",
                Severity.HIGH,
                node,
                f"accesses '{node.attr}', a sandbox-escape pattern",
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = _call_name(node)
        dotted = _dotted(node.func)
        if isinstance(node.func, ast.Name) and name in DANGEROUS_CALLS:
            self.add(
                "security.dangerous_call",
                "security",
                Severity.HIGH,
                node,
                f"calls '{name}()', not allowed in strategy code",
            )
        if name in SHIFT_LIKE and isinstance(node.func, ast.Attribute):
            self._check_shift(node, node.args[0] if node.args else _kw(node, "periods"), name)
        if dotted in ("np.roll", "numpy.roll"):
            amount = node.args[1] if len(node.args) > 1 else _kw(node, "shift")
            self._check_shift(node, amount, "np.roll")
        if name == "rolling" and _is_true(_kw(node, "center")):
            self.add(
                "lookahead.centered_window",
                "lookahead",
                Severity.HIGH,
                node,
                "centred rolling window averages future observations into each bar",
            )
        if name in ("bfill", "backfill"):
            self.add(
                "lookahead.backfill",
                "lookahead",
                Severity.HIGH,
                node,
                "back-filling copies future values into the past",
            )
        if name in ("fillna", "reindex") and _str_const(_kw(node, "method")) in (
            "bfill",
            "backfill",
        ):
            self.add(
                "lookahead.backfill",
                "lookahead",
                Severity.HIGH,
                node,
                f"{name}(method='bfill') copies future values into the past",
            )
        if name == "interpolate":
            self.add(
                "lookahead.backfill",
                "lookahead",
                Severity.MEDIUM,
                node,
                "interpolation uses the next observation to fill earlier gaps",
            )
        if name in AGGREGATIONS and isinstance(node.func, ast.Attribute):
            self._check_aggregation(node, name)
        if name in FIT and isinstance(node.func, ast.Attribute):
            data = node.args[0] if node.args else None
            if data is not None and not isinstance(data, ast.Subscript):
                self.add(
                    "lookahead.full_sample_fit",
                    "lookahead",
                    Severity.HIGH,
                    node,
                    f".{name}() on unsliced data fits on the full sample, including the future "
                    "(verified dynamically)",
                )
        self._check_random(node, name, dotted)
        self.generic_visit(node)

    def _check_shift(self, node: ast.Call, amount: ast.expr | None, what: str) -> None:
        if amount is None:
            return
        val = _eval(amount, self.env)
        if val is not None and val < 0:
            self.add(
                "lookahead.negative_shift",
                "lookahead",
                Severity.HIGH,
                node,
                f"{what} by {val:g} pulls future values into the current bar "
                "(verified dynamically)",
            )
        elif val is None and isinstance(amount, ast.UnaryOp) and isinstance(amount.op, ast.USub):
            self.add(
                "lookahead.negative_shift",
                "lookahead",
                Severity.HIGH,
                node,
                f"{what} by a negated amount is almost always a forward shift "
                "(verified dynamically)",
            )

    def _check_aggregation(self, node: ast.Call, name: str) -> None:
        assert isinstance(node.func, ast.Attribute)
        axis = _kw(node, "axis")
        if (isinstance(axis, ast.Constant) and axis.value in (1, "columns")) or (
            node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == 1
            and name != "quantile"
        ):
            return
        receiver_calls = set(_chain_calls(node.func.value))
        if receiver_calls & WINDOWED:
            return
        receiver = node.func.value
        if isinstance(receiver, ast.Subscript):  # row / column slices are not whole-history
            return
        if isinstance(receiver, ast.Name) and receiver.id in self.row_names:
            return
        if self._selects_single_row(receiver):
            return
        self.add(
            "lookahead.full_sample_stat",
            "lookahead",
            Severity.MEDIUM,
            node,
            f".{name}() over the whole history uses future data at every earlier bar "
            "(verified dynamically)",
        )

    def _check_random(self, node: ast.Call, name: str | None, dotted: str) -> None:
        if dotted.startswith(("np.random.", "numpy.random.")) and name in LEGACY_RANDOM:
            if not self.seeded_np:
                self.add(
                    "determinism.unseeded",
                    "determinism",
                    Severity.MEDIUM,
                    node,
                    f"np.random.{name}() without a seed makes results irreproducible",
                )
        elif dotted.endswith("default_rng") and not node.args and not node.keywords:
            self.add(
                "determinism.unseeded",
                "determinism",
                Severity.MEDIUM,
                node,
                "default_rng() without a seed makes results irreproducible",
            )
        elif dotted.startswith("random.") and name in LEGACY_RANDOM and not self.seeded_random:
            self.add(
                "determinism.unseeded",
                "determinism",
                Severity.MEDIUM,
                node,
                f"random.{name}() without a seed makes results irreproducible",
            )

    # ---------------------------------------------------------------- other patterns

    def visit_Subscript(self, node: ast.Subscript) -> None:
        sl = node.slice
        if isinstance(sl, ast.Slice) and sl.step is not None:
            step = _eval(sl.step, self.env)
            if step is not None and step < 0:
                self.add(
                    "lookahead.reverse_time",
                    "lookahead",
                    Severity.MEDIUM,
                    node,
                    "reversing the time axis lets later bars influence earlier ones",
                )
        if isinstance(node.value, ast.Attribute) and node.value.attr == "iloc":
            first = sl.elts[0] if isinstance(sl, ast.Tuple) and sl.elts else sl
            idx = _eval(first, self.env) if isinstance(first, ast.expr) else None
            if idx is not None and idx < 0:
                self.add(
                    "lookahead.end_of_sample",
                    "lookahead",
                    Severity.MEDIUM,
                    node,
                    ".iloc[-k] reads the end of the sample, which is the future for every "
                    "earlier bar unless used on point-in-time data",
                )
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, ast.Mult):
            left, right = node.left, node.right
            for ret, pos in ((left, right), (right, left)):
                if (
                    self._is_returns(ret)
                    and not self._is_lagged(pos)
                    and not self._is_lagged(ret)
                    and isinstance(pos, ast.Name | ast.Attribute | ast.Call | ast.Subscript)
                ):
                    self.add(
                        "execution.same_bar_pnl",
                        "execution",
                        Severity.MEDIUM,
                        node,
                        "positions multiplied by same-bar returns: P&L assumes a fill at "
                        "the signal bar (the engine fills at t+1)",
                    )
                    break
        self.generic_visit(node)

    def _is_returns(self, node: ast.expr) -> bool:
        if isinstance(node, ast.Name):
            return node.id in self.return_names
        return bool(set(_chain_calls(node)) & {"pct_change", "diff"})

    def _is_lagged(self, node: ast.expr) -> bool:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call) and _call_name(sub) == "shift":
                arg = sub.args[0] if sub.args else _kw(sub, "periods")
                val = _eval(arg, self.env) if arg is not None else 1.0
                if val is not None and val > 0:
                    return True
        return False

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str) and DATE_RE.match(node.value):
            self.add(
                "hindsight.hardcoded_date",
                "hindsight",
                Severity.LOW,
                node,
                f"hard-coded date '{node.value}': rules keyed to known events embed hindsight",
            )


@dataclass(frozen=True)
class ScanReport:
    findings: tuple[Finding, ...]
    parse_error: str | None = None

    def by_category(self, *cats: Category) -> list[Finding]:
        return [f for f in self.findings if f.category in cats]

    @property
    def max_severity(self) -> Severity:
        return max((f.severity for f in self.findings), default=Severity.INFO)

    def to_evidence(self, strategy_id: str) -> Evidence:
        if self.parse_error is not None:
            return make_evidence(
                TOOL,
                strategy_id,
                flag=False,
                severity=Severity.MEDIUM,
                summary=f"Source could not be parsed: {self.parse_error}.",
                details={"status": "inconclusive", "reason": self.parse_error},
                citations=CITATIONS,
            )
        serious = [f for f in self.findings if f.severity >= Severity.MEDIUM]
        rules = sorted({f.rule for f in serious})
        summary = (
            f"{len(serious)} suspicious pattern(s): " + ", ".join(rules) + "."
            if serious
            else "No suspicious patterns in the source."
        )
        return make_evidence(
            TOOL,
            strategy_id,
            flag=bool(serious),
            severity=min(self.max_severity, Severity.HIGH),
            statistic_name="n_findings_medium_or_worse",
            statistic=float(len(serious)),
            summary=summary,
            details={
                "status": "scanned",
                "findings": [
                    {
                        "rule": f.rule,
                        "category": f.category,
                        "severity": int(f.severity),
                        "line": f.line,
                        "snippet": f.snippet,
                        "message": f.message,
                    }
                    for f in self.findings
                ],
                "rules_triggered": rules,
            },
            citations=CITATIONS,
        )


def scan_source(source: str) -> ScanReport:
    source = textwrap.dedent(source)  # pasted snippets / nested defs arrive indented
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return ScanReport((), parse_error=f"{exc.msg} (line {exc.lineno})")
    scanner = _Scanner(source, tree)
    scanner.visit(tree)
    ordered = sorted(scanner.findings, key=lambda f: (f.line, f.col, f.rule))
    return ScanReport(tuple(ordered))


def security_findings(source: str) -> list[Finding]:
    """What the sandbox pre-check enforces before any untrusted code runs."""
    report = scan_source(source)
    if report.parse_error:
        return [
            Finding("security.parse_error", "security", Severity.HIGH, 0, 0, "", report.parse_error)
        ]
    return report.by_category("security")


def rules_triggered(findings: Iterable[Finding]) -> set[str]:
    return {f.rule for f in findings}
