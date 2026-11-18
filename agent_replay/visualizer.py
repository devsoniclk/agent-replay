"""
agent_replay.visualizer

Standalone interactive HTML dossier visualizer for agent flight recordings,
rendering timelines, state trees, tool call graphs, and diffs without external dependencies.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Optional, Union

from agent_replay.types import FlightRecording


class HTMLVisualizer:
    """Renders standalone dark-themed interactive HTML reports for flight recordings."""

    @classmethod
    def render(cls, recording: FlightRecording, output_path: Optional[Union[str, Path]] = None) -> str:
        """Render complete flight recording as self-contained HTML."""
        total_p, total_c, total_t = recording.total_tokens()
        latency = recording.total_latency_ms()
        tools_freq = recording.tool_usage_frequency()

        steps_html = []
        for s in recording.steps:
            st_val = s.step_type.value if hasattr(s.step_type, "value") else str(s.step_type)
            color_badge = {
                "thought": "#a855f7",
                "tool_call": "#3b82f6",
                "tool_result": "#10b981",
                "observation": "#f59e0b",
                "user_input": "#06b6d4",
                "final_response": "#22c55e",
                "error": "#ef4444",
                "state_mutation": "#ec4899",
            }.get(st_val, "#6b7280")

            tool_info = f"<code>{html.escape(s.tool_name)}</code>" if s.tool_name else ""
            content_preview = html.escape(s.content[:300] + ("..." if len(s.content) > 300 else ""))
            
            raw_json = html.escape(json.dumps(s.to_dict(), indent=2, default=str))

            step_card = f"""
            <div class="step-card" id="step-{s.step_index}">
                <div class="step-header">
                    <span class="step-index">#{s.step_index}</span>
                    <span class="step-badge" style="background: {color_badge};">{html.escape(st_val.upper())}</span>
                    {f'<span class="tool-tag">{tool_info}</span>' if tool_info else ''}
                    <span class="step-meta">{s.tokens_prompt + s.tokens_completion} tokens &bull; {s.latency_ms:.1f}ms</span>
                </div>
                <div class="step-body">
                    {f'<p class="step-text">{content_preview}</p>' if content_preview else ''}
                    <details class="json-details">
                        <summary>View Step Data JSON</summary>
                        <pre><code>{raw_json}</code></pre>
                    </details>
                </div>
            </div>
            """
            steps_html.append(step_card)

        tools_summary_rows = "".join(
            f"<tr><td><code>{html.escape(t)}</code></td><td>{c}</td></tr>"
            for t, c in sorted(tools_freq.items(), key=lambda x: -x[1])
        ) or "<tr><td colspan='2'>No tools invoked</td></tr>"

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Agent Replay &bull; {html.escape(recording.recording_id)}</title>
    <style>
        :root {{
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --bg-tertiary: #334155;
            --text-primary: #f8fafc;
            --text-muted: #94a3b8;
            --accent-cyan: #38bdf8;
            --accent-purple: #c084fc;
            --accent-green: #4ade80;
            --accent-red: #f87171;
            --border: #334155;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            padding: 24px;
            line-height: 1.5;
        }}
        .header {{
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
        }}
        .header h1 {{ font-size: 1.5rem; margin-bottom: 8px; color: var(--accent-cyan); }}
        .header p {{ color: var(--text-muted); font-size: 0.9rem; }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 16px;
            margin-top: 20px;
        }}
        .stat-card {{
            background: var(--bg-tertiary);
            padding: 16px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-value {{ font-size: 1.3rem; font-weight: bold; color: var(--text-primary); }}
        .stat-label {{ font-size: 0.8rem; color: var(--text-muted); text-transform: uppercase; margin-top: 4px; }}
        .container {{ display: grid; grid-template-columns: 2fr 1fr; gap: 24px; }}
        @media (max-width: 900px) {{ .container {{ grid-template-columns: 1fr; }} }}
        .timeline {{ display: flex; flexDirection: column; gap: 16px; }}
        .step-card {{
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 16px;
            transition: border-color 0.2s;
        }}
        .step-card:hover {{ border-color: var(--accent-cyan); }}
        .step-header {{ display: flex; align-items: center; gap: 10px; margin-bottom: 8px; flex-wrap: wrap; }}
        .step-index {{ font-weight: bold; color: var(--accent-cyan); }}
        .step-badge {{
            color: #fff;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
        }}
        .tool-tag code {{
            background: rgba(59, 130, 246, 0.2);
            color: #93c5fd;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.85rem;
        }}
        .step-meta {{ margin-left: auto; font-size: 0.8rem; color: var(--text-muted); }}
        .step-text {{ font-size: 0.9rem; color: #cbd5e1; white-space: pre-wrap; margin-bottom: 10px; }}
        .json-details summary {{
            font-size: 0.8rem;
            cursor: pointer;
            color: var(--accent-cyan);
            user-select: none;
        }}
        .json-details pre {{
            margin-top: 8px;
            background: #090d16;
            padding: 12px;
            border-radius: 6px;
            overflow-x: auto;
            font-size: 0.8rem;
            color: #38bdf8;
        }}
        .sidebar {{ display: flex; flex-direction: column; gap: 20px; }}
        .panel {{
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 20px;
        }}
        .panel h2 {{ font-size: 1.1rem; margin-bottom: 12px; color: var(--text-primary); border-bottom: 1px solid var(--border); padding-bottom: 8px; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
        th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid var(--border); }}
        th {{ color: var(--text-muted); }}
        .merkle-box {{ word-break: break-all; font-family: monospace; font-size: 0.75rem; background: #090d16; padding: 8px; border-radius: 4px; color: var(--accent-purple); }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Flight Recording Dossier &bull; {html.escape(recording.recording_id)}</h1>
        <p>Agent: <strong>{html.escape(recording.agent_name)}</strong> &bull; Framework: <strong>{html.escape(recording.framework)}</strong> &bull; Status: <strong>{html.escape(str(recording.status.value if hasattr(recording.status, 'value') else recording.status))}</strong> &bull; Branch: <strong>{html.escape(recording.branch_name)}</strong></p>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{len(recording.steps)}</div>
                <div class="stat-label">Total Steps</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{total_t:,}</div>
                <div class="stat-label">Total Tokens</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{latency:.1f}ms</div>
                <div class="stat-label">Cumulative Latency</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{len(tools_freq)}</div>
                <div class="stat-label">Unique Tools</div>
            </div>
        </div>
    </div>

    <div class="container">
        <div class="timeline">
            <h2>Execution Timeline</h2>
            {''.join(steps_html)}
        </div>
        <div class="sidebar">
            <div class="panel">
                <h2>Tool Call Frequency</h2>
                <table>
                    <thead><tr><th>Tool Name</th><th>Invocations</th></tr></thead>
                    <tbody>{tools_summary_rows}</tbody>
                </table>
            </div>
            <div class="panel">
                <h2>Cryptographic Proof</h2>
                <p style="font-size: 0.8rem; color: var(--text-muted); margin-bottom: 8px;">Merkle Action Tree Root Digest:</p>
                <div class="merkle-box">{html.escape(recording.merkle_root or recording.compute_merkle_root())}</div>
                <p style="font-size: 0.8rem; color: var(--text-muted); margin-top: 12px; margin-bottom: 4px;">Created At: {html.escape(recording.created_at)}</p>
            </div>
        </div>
    </div>
</body>
</html>"""

        if output_path:
            p = Path(output_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                f.write(html_content)

        return html_content

    @classmethod
    def export_html(cls, recording: FlightRecording, output_path: Union[str, Path]) -> str:
        """Alias for render with output_path."""
        return cls.render(recording, output_path=output_path)
