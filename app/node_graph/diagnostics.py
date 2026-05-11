"""Helpers for formatting node-graph diagnostics for UI/log/status surfaces."""

from __future__ import annotations

from collections import Counter, defaultdict
from html import escape
import re
from typing import Callable

from .engine import GraphDiagnostic


_CODE_LABEL_KEYS: dict[str, str] = {
    "NG001": "graph_diag_code_NG001",
    "NG002": "graph_diag_code_NG002",
    "NG003": "graph_diag_code_NG003",
    "NG004": "graph_diag_code_NG004",
    "NG005": "graph_diag_code_NG005",
    "NG006": "graph_diag_code_NG006",
    "NG007": "graph_diag_code_NG007",
    "NG008": "graph_diag_code_NG008",
    "NG009": "graph_diag_code_NG009",
    "NG010": "graph_diag_code_NG010",
    "NG011": "graph_diag_code_NG011",
}

_RULE_LABEL_KEYS: dict[str, str] = {
    "node_spec_exists": "graph_diag_rule_node_spec_exists",
    "edge_source_exists": "graph_diag_rule_edge_source_exists",
    "edge_destination_exists": "graph_diag_rule_edge_destination_exists",
    "downstream_allowed": "graph_diag_rule_downstream_allowed",
    "upstream_allowed": "graph_diag_rule_upstream_allowed",
    "source_port_exists": "graph_diag_rule_source_port_exists",
    "destination_port_exists": "graph_diag_rule_destination_port_exists",
    "port_types_compatible": "graph_diag_rule_port_types_compatible",
    "single_connection_per_input_port": "graph_diag_rule_single_connection_per_input_port",
    "required_input_connected": "graph_diag_rule_required_input_connected",
    "acyclic_graph": "graph_diag_rule_acyclic_graph",
}


def diagnostic_label(tr: Callable[[str], str], code: str) -> str:
    key = _CODE_LABEL_KEYS.get(str(code or "").strip().upper(), "")
    if not key:
        return str(code or "").strip().upper()
    text = tr(key)
    return text if text != key else str(code or "").strip().upper()


def diagnostic_primary_node_id(diagnostic: GraphDiagnostic) -> str:
    for value in (diagnostic.node_id, diagnostic.dst_node_id, diagnostic.src_node_id):
        node_id = str(value or "").strip()
        if node_id:
            return node_id
    return ""


def diagnostic_rule_label(tr: Callable[[str], str], rule: str) -> str:
    raw = str(rule or "").strip()
    if not raw:
        return ""
    key = _RULE_LABEL_KEYS.get(raw, "")
    if not key:
        return raw
    text = tr(key)
    return text if text != key else raw


def _port_label(tr: Callable[[str], str], port_name: str) -> str:
    raw = str(port_name or "").strip()
    if not raw:
        return ""
    key = f"graph_diag_port_{raw.lower()}"
    text = tr(key)
    if text == key:
        return raw
    return text


def _parse_missing_port(message: str, *, output_side: bool) -> tuple[str, str] | None:
    text = str(message or "").strip()
    if output_side:
        pattern = r"has no output port '([^']+)'\. Available: \[(.*)\]"
    else:
        pattern = r"has no input port '([^']+)'\. Available: \[(.*)\]"
    match = re.search(pattern, text)
    if not match:
        return None
    missing = str(match.group(1) or "").strip()
    available_raw = str(match.group(2) or "").strip()
    available = available_raw.replace("'", "")
    return missing, available


def _parse_type_mismatch(message: str) -> tuple[str, str, str, str] | None:
    text = str(message or "").strip()
    pattern = r"between '([^']+)' \(([^)]+)\) and '([^']+)' \(([^)]+)\)"
    match = re.search(pattern, text)
    if not match:
        return None
    src_ref = str(match.group(1) or "").strip()
    src_type = str(match.group(2) or "").strip()
    dst_ref = str(match.group(3) or "").strip()
    dst_type = str(match.group(4) or "").strip()
    return src_ref, src_type, dst_ref, dst_type


def _node_ref(node_id: str, node_label_for_id: Callable[[str], str] | None = None) -> str:
    raw = str(node_id or "").strip()
    if not raw:
        return ""
    if not callable(node_label_for_id):
        return raw
    label = str(node_label_for_id(raw) or "").strip()
    if not label or label == raw:
        return raw
    return f"{label} [{raw}]"


def format_graph_diagnostic_context(
    tr: Callable[[str], str],
    diagnostic: GraphDiagnostic,
    *,
    node_label_for_id: Callable[[str], str] | None = None,
) -> str:
    code = str(diagnostic.code or "").strip().upper()
    parts: list[str] = []
    if diagnostic.node_id:
        parts.append(
            tr("graph_diag_context_node").format(
                node_id=_node_ref(str(diagnostic.node_id), node_label_for_id)
            )
        )

    src_node_ref = _node_ref(str(diagnostic.src_node_id or "").strip(), node_label_for_id)
    src_ref = f"{src_node_ref}.{diagnostic.src_port}" if (src_node_ref and diagnostic.src_port) else src_node_ref
    dst_node_ref = _node_ref(str(diagnostic.dst_node_id or "").strip(), node_label_for_id)
    dst_ref = f"{dst_node_ref}.{diagnostic.dst_port}" if (dst_node_ref and diagnostic.dst_port) else dst_node_ref
    if src_ref or dst_ref:
        edge_key = "graph_diag_context_edge_blocked" if code in {"NG004", "NG005"} else "graph_diag_context_edge"
        parts.append(
            tr(edge_key).format(
                src=src_ref or "?",
                dst=dst_ref or "?",
            )
        )
    elif diagnostic.dst_port:
        if code == "NG010":
            port_name = str(diagnostic.dst_port or "").strip()
            parts.append(
                tr("graph_diag_required_input_missing").format(
                    port=_port_label(tr, port_name),
                    port_name=port_name,
                )
            )
        else:
            parts.append(tr("graph_diag_context_port").format(port=diagnostic.dst_port))

    if diagnostic.rule:
        rule_text = diagnostic_rule_label(tr, diagnostic.rule)
        if code in {"NG004", "NG005"}:
            parts.append(tr("graph_diag_context_reason").format(reason=rule_text))
        else:
            parts.append(tr("graph_diag_context_rule").format(rule=rule_text))

    if code == "NG006":
        parsed = _parse_missing_port(diagnostic.message, output_side=True)
        if parsed is not None:
            missing, available = parsed
            parts.append(
                tr("graph_diag_missing_output_port_detail").format(
                    port=missing,
                    available=available or "-",
                )
            )
    elif code == "NG007":
        parsed = _parse_missing_port(diagnostic.message, output_side=False)
        if parsed is not None:
            missing, available = parsed
            parts.append(
                tr("graph_diag_missing_input_port_detail").format(
                    port=missing,
                    available=available or "-",
                )
            )
    elif code == "NG008":
        parsed = _parse_type_mismatch(diagnostic.message)
        if parsed is not None:
            src_ref_raw, src_type, dst_ref_raw, dst_type = parsed
            parts.append(
                tr("graph_diag_type_mismatch_detail").format(
                    src=src_ref_raw,
                    src_type=src_type,
                    dst=dst_ref_raw,
                    expected=dst_type,
                    actual=src_type,
                )
            )

    return " | ".join(part for part in parts if part)


def format_graph_diagnostics_summary(tr: Callable[[str], str], diagnostics: list[GraphDiagnostic]) -> str:
    if not diagnostics:
        return tr("graph_diagnostics_status_ok")
    counts = Counter(str(diag.code or "").strip().upper() for diag in diagnostics)
    codes = ", ".join(f"{code}x{counts[code]}" for code in sorted(counts))
    return tr("graph_diagnostics_status_errors").format(count=len(diagnostics), codes=codes)


def format_graph_diagnostics_text(
    tr: Callable[[str], str],
    diagnostics: list[GraphDiagnostic],
    *,
    node_label_for_id: Callable[[str], str] | None = None,
) -> str:
    if not diagnostics:
        return tr("graph_diagnostics_ok")

    grouped: dict[str, list[GraphDiagnostic]] = defaultdict(list)
    for diagnostic in diagnostics:
        grouped[str(diagnostic.code or "").strip().upper()].append(diagnostic)

    lines = [format_graph_diagnostics_summary(tr, diagnostics)]
    for code in sorted(grouped):
        items = grouped[code]
        lines.append(
            tr("graph_diagnostics_group_line").format(
                code=code,
                label=diagnostic_label(tr, code),
                count=len(items),
            )
        )
        for diagnostic in items:
            detail = (
                format_graph_diagnostic_context(
                    tr,
                    diagnostic,
                    node_label_for_id=node_label_for_id,
                )
                or diagnostic.message
            )
            lines.append(tr("graph_diagnostics_entry_line").format(detail=detail))
    return "\n".join(line for line in lines if line)


def format_graph_diagnostics_html(
    tr: Callable[[str], str],
    diagnostics: list[GraphDiagnostic],
    *,
    node_label_for_id: Callable[[str], str] | None = None,
    target_for_diagnostic: Callable[[GraphDiagnostic], str] | None = None,
) -> str:
    if not diagnostics:
        return (
            f'<span style="color:#7edc9a;font-weight:600;">{escape(tr("graph_diagnostics_ok"))}</span>'
        )

    grouped: dict[str, list[GraphDiagnostic]] = defaultdict(list)
    for diagnostic in diagnostics:
        grouped[str(diagnostic.code or "").strip().upper()].append(diagnostic)

    parts = [
        f'<div style="color:#f0b35c;font-weight:700;margin-bottom:6px;">'
        f'{escape(format_graph_diagnostics_summary(tr, diagnostics))}</div>'
    ]
    for code in sorted(grouped):
        items = grouped[code]
        heading = tr("graph_diagnostics_group_line").format(
            code=code,
            label=diagnostic_label(tr, code),
            count=len(items),
        )
        parts.append(f'<div style="color:#dfe8f4;font-weight:600;margin-top:6px;">{escape(heading)}</div>')
        for diagnostic in items:
            detail = (
                format_graph_diagnostic_context(
                    tr,
                    diagnostic,
                    node_label_for_id=node_label_for_id,
                )
                or diagnostic.message
            )
            detail_html = escape(detail)
            if callable(target_for_diagnostic):
                target = str(target_for_diagnostic(diagnostic) or "").strip()
                if target:
                    detail_html = f'<a href="{escape(target, quote=True)}" style="color:#9fd8ff; text-decoration:none;">{detail_html}</a>'
            parts.append(f'<div style="color:#9fb2c8;margin-left:10px;">&bull; {detail_html}</div>')
    return "".join(parts)