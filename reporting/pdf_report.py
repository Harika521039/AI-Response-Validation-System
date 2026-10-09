
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence
from analytics import dashboard
MAX_FIELD_CHARS = 6000
TRUNCATION_NOTICE = " [text truncated for the report]"
NOT_AVAILABLE = "n/a"
class PdfRenderingUnavailable(RuntimeError):
    """Raised when ReportLab is not installed, so the caller can say so plainly."""
def _clip(text: Any, limit: int = MAX_FIELD_CHARS) -> str:
    value = "" if text is None else str(text)
    value = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    if len(value) <= limit:
        return value
    return value[: limit - len(TRUNCATION_NOTICE)].rstrip() + TRUNCATION_NOTICE
def _number(value: Any, suffix: str = "") -> str:
    if value is None:
        return NOT_AVAILABLE
    if isinstance(value, float):
        return f"{value:g}{suffix}"
    return f"{value}{suffix}"
@dataclass
class Heading:
    text: str
    level: int = 1
@dataclass
class Paragraph:
    text: str
@dataclass
class BulletList:
    items: List[str] = field(default_factory=list)
@dataclass
class Table:
    columns: List[str]
    rows: List[List[str]]
    title: str = ""
@dataclass
class BarChart:
    title: str
    categories: List[str]
    values: List[float]
    value_label: str = ""
@dataclass
class PageBreak:
    pass
Block = Any
@dataclass
class ReportDocument:
    """The finished report, before it becomes PDF bytes."""
    title: str
    subtitle: str = ""
    blocks: List[Block] = field(default_factory=list)
    def add(self, block: Block) -> "ReportDocument":
        self.blocks.append(block)
        return self
    def headings(self) -> List[str]:
        return [b.text for b in self.blocks if isinstance(b, Heading)]
    def tables(self) -> List[Table]:
        return [b for b in self.blocks if isinstance(b, Table)]
    def charts(self) -> List[BarChart]:
        return [b for b in self.blocks if isinstance(b, BarChart)]
    def to_text(self) -> str:
        """Everything the report says, as plain text, for verification."""
        lines: List[str] = [self.title]
        if self.subtitle:
            lines.append(self.subtitle)
        for block in self.blocks:
            if isinstance(block, Heading):
                lines.append(block.text)
            elif isinstance(block, Paragraph):
                lines.append(block.text)
            elif isinstance(block, BulletList):
                lines.extend(block.items)
            elif isinstance(block, Table):
                if block.title:
                    lines.append(block.title)
                lines.append(" | ".join(block.columns))
                for row in block.rows:
                    lines.append(" | ".join(str(cell) for cell in row))
            elif isinstance(block, BarChart):
                lines.append(block.title)
                for category, value in zip(block.categories, block.values):
                    lines.append(f"{category}: {value}")
        return "\n".join(lines)
def _metadata_table(
    records: Sequence[dashboard.EvaluationRecord],
    stats: Dict[str, Any],
    generated_at: str,
    scope: str,
) -> Table:
    batches = stats["batch_statistics"]
    period = [record.created_at for record in records if record.created_at]
    rows = [
        ["Report generated (UTC)", generated_at],
        ["Scope", scope],
        ["Responses included", str(stats["total_responses"])],
        ["Batches covered", str(stats["batches_covered"])],
        ["Batch labels", ", ".join(item["batch_label"] for item in batches) or NOT_AVAILABLE],
        ["Earliest evaluation", min(period) if period else NOT_AVAILABLE],
        ["Latest evaluation", max(period) if period else NOT_AVAILABLE],
    ]
    return Table(columns=["Field", "Value"], rows=rows, title="Report metadata")
def _summary_table(stats: Dict[str, Any]) -> Table:
    rows = [
        ["Total responses", str(stats["total_responses"])],
        ["Responses with a final verdict", str(stats["responses_with_verdict"])],
        ["Average overall score (0-100)", _number(stats["average_overall_score"])],
        ["Highest overall score", _number(stats["highest_overall_score"])],
        ["Lowest overall score", _number(stats["lowest_overall_score"])],
    ]
    return Table(columns=["Metric", "Value"], rows=rows, title="Batch summary")
def _verdict_table(stats: Dict[str, Any]) -> Table:
    rows = []
    for item in stats["verdict_distribution"]:
        rows.append(
            [
                item["Verdict"],
                str(item["Count"]),
                _number(item["Percent"], "%"),
            ]
        )
    return Table(columns=["Verdict", "Count", "Share"], rows=rows, title="Verdict counts")
def _dimension_table(stats: Dict[str, Any]) -> Table:
    normalized = stats["average_normalized_scores"]
    rows = [
        ["Relevance", _number(stats["average_relevance"], "/5"), _number(normalized.get("relevance"))],
        ["Accuracy", _number(stats["average_accuracy"], "/5"), _number(normalized.get("accuracy"))],
        ["Completeness", _number(stats["average_completeness"], "/5"), _number(normalized.get("completeness"))],
        ["Hallucination", NOT_AVAILABLE, _number(normalized.get("hallucination"))],
    ]
    return Table(
        columns=["Dimension", "Average judge score", "Average normalized (0-100)"],
        rows=rows,
        title="Average dimension scores",
    )
def _hallucination_table(stats: Dict[str, Any]) -> Table:
    halluc = stats["hallucination"]
    counts = halluc["status_counts"]
    rows = [
        ["Responses judged for hallucination", str(halluc["responses_judged"])],
        ["No hallucination", str(counts.get("No hallucination", 0))],
        ["Partially hallucinated", str(counts.get("Partially hallucinated", 0))],
        ["Hallucinated", str(counts.get("Hallucinated", 0))],
        ["Hallucination frequency", _number(halluc["hallucination_frequency_percent"], "%")],
        ["Average hallucination score (0-100)", _number(halluc["average_hallucination_score"])],
        ["Claims checked", str(halluc["claims_checked"])],
        ["Unsupported claims", str(halluc["unsupported_claims"])],
        ["Contradicted claims", str(halluc["contradicted_claims"])],
        ["Share of claims flagged", _number(halluc["problem_claim_percent"], "%")],
    ]
    return Table(columns=["Hallucination statistic", "Value"], rows=rows, title="Hallucination statistics")
def _batch_table(stats: Dict[str, Any]) -> Table:
    rows = []
    for item in stats["batch_statistics"]:
        rows.append(
            [
                item["batch_label"],
                str(item["total_responses"]),
                _number(item["average_overall_score"]),
                _number(item["pass_rate_percent"], "%"),
                _number(item["hallucination_frequency_percent"], "%"),
            ]
        )
    return Table(
        columns=["Batch", "Responses", "Average score", "Pass rate", "Hallucination rate"],
        rows=rows,
        title="Per-batch statistics",
    )
def _issues_table(stats: Dict[str, Any]) -> Table:
    rows = []
    for item in stats["frequent_issues"]:
        rows.append(
            [
                item["category"],
                _clip(item["issue"], 400),
                str(item["count"]),
                _number(item["affected_percent"], "%"),
            ]
        )
    return Table(
        columns=["Category", "Issue", "Occurrences", "Share of responses"],
        rows=rows,
        title="Most frequent evaluation issues",
    )
def _results_table(records: Sequence[dashboard.EvaluationRecord]) -> Table:
    rows = []
    for record in records:
        rows.append(
            [
                _clip(record.record_id, 60),
                _clip(record.question, 160),
                _number(record.relevance_score),
                _number(record.accuracy_score),
                _number(record.completeness_score),
                _number(record.overall_score),
                record.verdict or NOT_AVAILABLE,
            ]
        )
    return Table(
        columns=["ID", "Question", "Rel", "Acc", "Comp", "Overall", "Verdict"],
        rows=rows,
        title="All evaluated responses",
    )
def _add_record_detail(document: ReportDocument, record: dashboard.EvaluationRecord, index: int) -> None:
    document.add(Heading(f"Result {index}: {record.record_id}", level=2))
    document.add(Paragraph(f"Source: {record.group_label}"))
    if record.created_at:
        document.add(Paragraph(f"Evaluated at: {record.created_at}"))
    document.add(Heading("Question", level=3))
    document.add(Paragraph(_clip(record.question)))
    document.add(Heading("AI response", level=3))
    document.add(Paragraph(_clip(record.response)))
    if record.reference_answer:
        document.add(Heading("Reference answer", level=3))
        document.add(Paragraph(_clip(record.reference_answer)))
    if record.source_information:
        document.add(Heading("Source information supplied", level=3))
        document.add(Paragraph(_clip(record.source_information)))
    document.add(
        Table(
            columns=["Dimension", "Result"],
            rows=[
                ["Relevance", _number(record.relevance_score, "/5")],
                ["Accuracy", _number(record.accuracy_score, "/5")],
                ["Completeness", _number(record.completeness_score, "/5")],
                ["Hallucination", record.hallucination_status or NOT_AVAILABLE],
                ["Weighted overall score", _number(record.overall_score, "/100")],
                ["Final verdict", record.verdict or NOT_AVAILABLE],
            ],
            title="Scores and verdict",
        )
    )
    document.add(Heading("Agent reasoning", level=3))
    reasoning_rows = []
    for label, section in (
        ("Relevance judge", record.relevance),
        ("Accuracy judge", record.accuracy),
        ("Completeness judge", record.completeness),
        ("Hallucination detector", record.hallucination),
        ("Verdict agent", record.verdict_section),
    ):
        text = section.get("reasoning") if isinstance(section, dict) else None
        if text:
            reasoning_rows.append([label, _clip(text, 3000)])
    if reasoning_rows:
        document.add(Table(columns=["Agent", "Reasoning"], rows=reasoning_rows))
    else:
        document.add(Paragraph("No agent reasoning was recorded for this result."))
    evidence = record.accuracy.get("supporting_evidence")
    document.add(Heading("Supporting evidence used", level=3))
    if isinstance(evidence, list) and evidence:
        document.add(BulletList([_clip(item, 1200) for item in evidence if item]))
    else:
        document.add(Paragraph("No supporting evidence was available for this response."))
    retrieved = record.payload.get("retrieved_evidence")
    if isinstance(retrieved, list) and retrieved:
        document.add(Heading("Retrieved reference evidence", level=3))
        document.add(
            Table(
                columns=["Source", "Evidence"],
                rows=[
                    [_clip(item.get("source", "unknown"), 80), _clip(item.get("text", ""), 1500)]
                    for item in retrieved
                    if isinstance(item, dict)
                ],
            )
        )
    document.add(Heading("Hallucinated or unsupported claims", level=3))
    problems = record.problem_claims
    if problems:
        document.add(
            Table(
                columns=["Claim", "Status", "Evidence", "Reasoning"],
                rows=[
                    [
                        _clip(claim.get("claim", ""), 900),
                        claim.get("claim_status", NOT_AVAILABLE),
                        _clip(claim.get("evidence") or "None available", 900),
                        _clip(claim.get("reasoning", ""), 900),
                    ]
                    for claim in problems
                ],
            )
        )
    else:
        document.add(Paragraph("No claim in this response was unsupported or contradicted."))
    document.add(Heading("Completeness: missing aspects", level=3))
    if record.missing_aspects:
        document.add(BulletList([_clip(aspect, 600) for aspect in record.missing_aspects]))
    else:
        document.add(Paragraph("No missing aspects were identified for this response."))
    if record.partial_aspects:
        document.add(Heading("Completeness: partially addressed aspects", level=3))
        document.add(BulletList([_clip(aspect, 600) for aspect in record.partial_aspects]))
    if record.major_issues:
        document.add(Heading("Major issues", level=3))
        document.add(BulletList([_clip(issue, 600) for issue in record.major_issues]))
def build_report_document(
    records: Sequence[dashboard.EvaluationRecord],
    title: str = "AI Response Validation Report",
    scope: str = "All stored evaluations",
    generated_at: Optional[str] = None,
    include_details: bool = True,
    max_detail_records: Optional[int] = None,
) -> ReportDocument:
    """
    Assemble the full M4.2 report from the supplied evaluation records.
    Every figure comes from dashboard.compute_statistics(), which reads the
    agents' own stored output. With no records the document still renders,
    saying explicitly that there was nothing to report.
    """
    records = list(records)
    timestamp = generated_at or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    document = ReportDocument(
        title=title,
        subtitle=f"Generated {timestamp} UTC from {len(records)} stored evaluation results",
    )
    if not records:
        document.add(Heading("No evaluation data", level=1))
        document.add(
            Paragraph(
                "There are no completed evaluation results in the selected scope, "
                "so this report contains no statistics. Run an evaluation and export again."
            )
        )
        return document
    stats = dashboard.compute_statistics(records)
    document.add(Heading("Report metadata", level=1))
    document.add(_metadata_table(records, stats, timestamp, scope))
    document.add(Heading("Batch summary", level=1))
    document.add(_summary_table(stats))
    document.add(Heading("Verdict distribution", level=1))
    document.add(_verdict_table(stats))
    distribution = stats["verdict_distribution"]
    document.add(
        BarChart(
            title="Responses per verdict",
            categories=[item["Verdict"] for item in distribution],
            values=[float(item["Count"]) for item in distribution],
            value_label="Responses",
        )
    )
    document.add(Heading("Average dimension scores", level=1))
    document.add(_dimension_table(stats))
    dimension_chart = stats["dimension_chart"]
    if dimension_chart:
        document.add(
            BarChart(
                title="Average normalized score per dimension",
                categories=[item["Dimension"] for item in dimension_chart],
                values=[float(item["Average score (0-100)"]) for item in dimension_chart],
                value_label="Score (0-100)",
            )
        )
    document.add(Heading("Hallucination statistics", level=1))
    document.add(_hallucination_table(stats))
    document.add(Heading("Per-batch statistics", level=1))
    document.add(_batch_table(stats))
    trend = [item for item in stats["quality_trend"] if item["Average overall score"] is not None]
    if len(trend) > 1:
        document.add(
            BarChart(
                title="Average overall score per batch",
                categories=[item["Batch"] for item in trend],
                values=[float(item["Average overall score"]) for item in trend],
                value_label="Score (0-100)",
            )
        )
    if stats["frequent_issues"]:
        document.add(Heading("Most frequent evaluation issues", level=1))
        document.add(_issues_table(stats))
    recommendations = dashboard.improvement_recommendations(records)
    document.add(Heading("Improvement recommendations", level=1))
    if recommendations:
        document.add(BulletList(recommendations))
    else:
        document.add(
            Paragraph(
                "Every response in this scope passed without flagged claims or missing aspects, "
                "so the evaluation data gives no basis for a recommendation."
            )
        )
    document.add(Heading("Individual results", level=1))
    document.add(_results_table(records))
    if include_details:
        detailed = records if max_detail_records is None else records[:max_detail_records]
        if max_detail_records is not None and len(records) > len(detailed):
            document.add(
                Paragraph(
                    f"Detailed sections are included for the first {len(detailed)} of "
                    f"{len(records)} results. The table above covers every result."
                )
            )
        for index, record in enumerate(detailed, start=1):
            document.add(PageBreak())
            _add_record_detail(document, record, index)
    return document
def _escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br/>")
    )
def render_pdf(document: ReportDocument) -> bytes:
    """
    Render a ReportDocument to PDF bytes with ReportLab.
    Raises
    ------
    PdfRenderingUnavailable
        If ReportLab is not installed in the running environment.
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_LEFT
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.graphics.charts.barcharts import VerticalBarChart
        from reportlab.graphics.shapes import Drawing
        from reportlab.platypus import (
            KeepTogether,
            ListFlowable,
            ListItem,
            PageBreak as PlatypusPageBreak,
            Paragraph as PlatypusParagraph,
            SimpleDocTemplate,
            Spacer,
            Table as PlatypusTable,
            TableStyle,
        )
    except ImportError as exc:
        raise PdfRenderingUnavailable(
            "PDF export needs the reportlab package. Install it with: pip install reportlab"
        ) from exc
    import io
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title=document.title,
        author="AI Response Validation System",
    )
    sheet = getSampleStyleSheet()
    body = ParagraphStyle(
        "ReportBody",
        parent=sheet["BodyText"],
        fontSize=9,
        leading=12.5,
        alignment=TA_LEFT,
        spaceAfter=5,
    )
    cell = ParagraphStyle("ReportCell", parent=body, fontSize=8, leading=10.5, spaceAfter=0)
    header_cell = ParagraphStyle("ReportHeaderCell", parent=cell, textColor=colors.white, fontName="Helvetica-Bold")
    heading_styles = {
        1: ParagraphStyle("H1", parent=sheet["Heading1"], fontSize=14, spaceBefore=10, spaceAfter=6,
                          textColor=colors.HexColor("#183153")),
        2: ParagraphStyle("H2", parent=sheet["Heading2"], fontSize=12, spaceBefore=8, spaceAfter=5,
                          textColor=colors.HexColor("#183153")),
        3: ParagraphStyle("H3", parent=sheet["Heading3"], fontSize=10, spaceBefore=6, spaceAfter=3,
                          textColor=colors.HexColor("#117a72")),
    }
    title_style = ParagraphStyle(
        "ReportTitle", parent=sheet["Title"], fontSize=18, leading=22,
        textColor=colors.HexColor("#172033"), spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle", parent=body, fontSize=9.5, textColor=colors.HexColor("#667085"), spaceAfter=12,
    )
    available_width = doc.width
    story: List[Any] = [PlatypusParagraph(_escape(document.title), title_style)]
    if document.subtitle:
        story.append(PlatypusParagraph(_escape(document.subtitle), subtitle_style))
    def build_table(block: Table) -> List[Any]:
        flowables: List[Any] = []
        if block.title:
            flowables.append(PlatypusParagraph(_escape(block.title), heading_styles[3]))
        header = [PlatypusParagraph(_escape(name), header_cell) for name in block.columns]
        data = [header]
        for row in block.rows:
            data.append([PlatypusParagraph(_escape(value), cell) for value in row])
        if len(block.rows) == 0:
            data.append([PlatypusParagraph("No rows.", cell)] + [PlatypusParagraph("", cell)] * (len(block.columns) - 1))
        column_count = max(1, len(block.columns))
        widths = [available_width / column_count] * column_count
        if column_count == 2:
            widths = [available_width * 0.34, available_width * 0.66]
        table = PlatypusTable(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#183153")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c9d2de")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f5f9")]),
                ]
            )
        )
        flowables.append(table)
        flowables.append(Spacer(1, 8))
        return flowables
    def build_chart(block: BarChart) -> List[Any]:
        width = available_width
        height = 150
        drawing = Drawing(width, height)
        chart = VerticalBarChart()
        chart.x = 40
        chart.y = 35
        chart.width = width - 70
        chart.height = height - 55
        chart.data = [[float(value) for value in block.values]]
        chart.categoryAxis.categoryNames = [str(name) for name in block.categories]
        chart.categoryAxis.labels.angle = 20 if max((len(str(c)) for c in block.categories), default=0) > 10 else 0
        chart.categoryAxis.labels.dy = -8
        chart.categoryAxis.labels.fontSize = 7
        chart.valueAxis.valueMin = 0
        top = max(block.values) if block.values else 0
        chart.valueAxis.valueMax = max(1.0, top * 1.15)
        chart.valueAxis.labels.fontSize = 7
        chart.bars[0].fillColor = colors.HexColor("#117a72")
        chart.barLabels.fontSize = 7
        chart.barLabelFormat = "%0.1f"
        chart.barLabels.dy = 4
        drawing.add(chart)
        return [
            PlatypusParagraph(_escape(block.title), heading_styles[3]),
            drawing,
            Spacer(1, 10),
        ]
    for block in document.blocks:
        if isinstance(block, Heading):
            story.append(PlatypusParagraph(_escape(block.text), heading_styles.get(block.level, heading_styles[3])))
        elif isinstance(block, Paragraph):
            story.append(PlatypusParagraph(_escape(block.text) or "&nbsp;", body))
        elif isinstance(block, BulletList):
            items = [ListItem(PlatypusParagraph(_escape(text), body), leftIndent=12) for text in block.items]
            if items:
                story.append(ListFlowable(items, bulletType="bullet", start="-", leftIndent=14))
                story.append(Spacer(1, 6))
        elif isinstance(block, Table):
            story.extend(build_table(block))
        elif isinstance(block, BarChart):
            story.append(KeepTogether(build_chart(block)))
        elif isinstance(block, PageBreak):
            story.append(PlatypusPageBreak())
    def decorate(canvas, current_doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#667085"))
        canvas.drawString(18 * mm, 10 * mm, "AI Response Validation System - evaluation report")
        canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {canvas.getPageNumber()}")
        canvas.restoreState()
    doc.build(story, onFirstPage=decorate, onLaterPages=decorate)
    return buffer.getvalue()
def generate_pdf(
    records: Sequence[dashboard.EvaluationRecord],
    title: str = "AI Response Validation Report",
    scope: str = "All stored evaluations",
    generated_at: Optional[str] = None,
    include_details: bool = True,
    max_detail_records: Optional[int] = None,
) -> bytes:
    """Build the report from real records and render it in one call."""
    document = build_report_document(
        records,
        title=title,
        scope=scope,
        generated_at=generated_at,
        include_details=include_details,
        max_detail_records=max_detail_records,
    )
    return render_pdf(document)
def is_pdf_available() -> bool:
    try:
        import reportlab
    except ImportError:
        return False
    return True
