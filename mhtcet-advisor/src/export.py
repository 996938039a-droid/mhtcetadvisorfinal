"""
export.py — PDF export of the preference list using ReportLab.
"""

import io
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Table, TableStyle,
    Spacer, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT


# Colour palette
COLOR_REACH  = colors.HexColor("#EF4444")
COLOR_DREAM  = colors.HexColor("#F97316")
COLOR_TARGET = colors.HexColor("#3B82F6")
COLOR_SAFE   = colors.HexColor("#22C55E")
COLOR_ASSURED= colors.HexColor("#6B7280")
COLOR_HEADER = colors.HexColor("#1E3A5F")
COLOR_LIGHT  = colors.HexColor("#F0F4FF")

LABEL_COLORS = {
    "Reach": COLOR_REACH,
    "Dream": COLOR_DREAM,
    "Target": COLOR_TARGET,
    "Safe": COLOR_SAFE,
    "Assured": COLOR_ASSURED,
}


def generate_pdf(student_profile: dict, preference_list) -> bytes:
    """
    Generate a PDF report of the preference list.

    Args:
        student_profile: dict with student details
        preference_list: pandas DataFrame (indexed from 1)

    Returns:
        PDF as bytes
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        rightMargin=1.5*cm, leftMargin=1.5*cm,
        topMargin=1.5*cm, bottomMargin=1.5*cm
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'Title', parent=styles['Title'],
        textColor=COLOR_HEADER, fontSize=16, spaceAfter=6
    )
    subtitle_style = ParagraphStyle(
        'Subtitle', parent=styles['Normal'],
        textColor=colors.HexColor("#555555"), fontSize=9, spaceAfter=4
    )
    section_style = ParagraphStyle(
        'Section', parent=styles['Heading2'],
        textColor=COLOR_HEADER, fontSize=11, spaceBefore=12, spaceAfter=4
    )
    note_style = ParagraphStyle(
        'Note', parent=styles['Normal'],
        fontSize=7.5, textColor=colors.HexColor("#666666"), leading=11
    )

    story = []

    # ── Header ────────────────────────────────────────────────────────────────
    story.append(Paragraph("MHT-CET College Preference List", title_style))
    story.append(Paragraph(
        f"Generated on {datetime.now().strftime('%d %B %Y, %I:%M %p')}",
        subtitle_style
    ))
    story.append(HRFlowable(width="100%", thickness=1, color=COLOR_HEADER, spaceAfter=8))

    # ── Student Profile ────────────────────────────────────────────────────────
    story.append(Paragraph("Student Profile", section_style))

    profile_data = [
        ["Percentile", str(student_profile.get("percentile", "—")),
         "Category", student_profile.get("category", "—")],
        ["Gender", student_profile.get("gender", "—").capitalize(),
         "Home District", student_profile.get("district", "—")],
        ["Home University", student_profile.get("home_university", "—"),
         "CAP Round", f"Round {student_profile.get('cap_round', 1)}"],
        ["Preferred Branches", ", ".join(student_profile.get("branches", [])) or "All",
         "Cutoff Adjustment", f"{student_profile.get('trend_adj', 0):+.1f} pct pts"],
    ]

    profile_table = Table(profile_data, colWidths=[3*cm, 5.5*cm, 3.5*cm, 5*cm])
    profile_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), COLOR_LIGHT),
        ('BACKGROUND', (2, 0), (2, -1), COLOR_LIGHT),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8.5),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#DDDDDD")),
        ('PADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(profile_table)
    story.append(Spacer(1, 8))

    # ── Preference List ────────────────────────────────────────────────────────
    story.append(Paragraph("Recommended Preference List", section_style))

    if preference_list is None or len(preference_list) == 0:
        story.append(Paragraph("No matching options found.", note_style))
    else:
        headers = ["#", "College Name", "Branch", "Category", "Pred. Cutoff", "Probability", "Class"]
        table_data = [headers]

        for idx, row in preference_list.iterrows():
            cl = row['classification']
            label = cl['label'] if isinstance(cl, dict) else cl
            prob = row.get('probability', 0)
            cutoff = row.get('predicted_cutoff', 0)
            table_data.append([
                str(idx),
                _truncate(row.get('college_name', ''), 38),
                _truncate(row.get('course_name', ''), 28),
                row.get('best_category', ''),
                f"{cutoff:.2f}",
                f"{prob:.0f}%",
                label,
            ])

        col_widths = [0.7*cm, 6.5*cm, 4.5*cm, 1.8*cm, 2*cm, 1.8*cm, 1.5*cm]
        pref_table = Table(table_data, colWidths=col_widths, repeatRows=1)

        base_style = [
            ('BACKGROUND', (0, 0), (-1, 0), COLOR_HEADER),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 7.5),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('GRID', (0, 0), (-1, -1), 0.3, colors.HexColor("#CCCCCC")),
            ('PADDING', (0, 0), (-1, -1), 3),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, COLOR_LIGHT]),
        ]

        # Colour the classification column per label
        for i, row in enumerate(preference_list.itertuples(), start=1):
            cl = row.classification
            label = cl['label'] if isinstance(cl, dict) else cl
            c = LABEL_COLORS.get(label, colors.black)
            base_style.append(('TEXTCOLOR', (6, i), (6, i), c))
            base_style.append(('FONTNAME', (6, i), (6, i), 'Helvetica-Bold'))

        pref_table.setStyle(TableStyle(base_style))
        story.append(pref_table)

    story.append(Spacer(1, 10))

    # ── Legend ────────────────────────────────────────────────────────────────
    story.append(Paragraph("Classification Legend", section_style))
    legend_data = [
        ["🎯 Reach (<10%)", "Very unlikely — include for hope"],
        ["⭐ Dream (10–30%)", "Ambitious — needs luck or rising in next round"],
        ["✅ Target (30–70%)", "Realistic — good chance of admission"],
        ["🛡️ Safe (70–90%)", "High probability of getting this seat"],
        ["🔒 Assured (>90%)", "Almost certain if preference is high enough"],
    ]
    legend_table = Table(legend_data, colWidths=[5*cm, 12*cm])
    legend_table.setStyle(TableStyle([
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 0.3, colors.HexColor("#DDDDDD")),
        ('BACKGROUND', (0, 0), (-1, -1), COLOR_LIGHT),
        ('PADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(legend_table)

    # ── Disclaimer ────────────────────────────────────────────────────────────
    story.append(Spacer(1, 12))
    story.append(Paragraph(
        "⚠️ Disclaimer: This preference list is generated using historical data from 2022–2024 and is for "
        "guidance purposes only. Actual cutoffs may vary. Always cross-check with the official Maharashtra "
        "State CET Cell website (fe2025.mahacet.org) before finalising your preference list.",
        note_style
    ))

    doc.build(story)
    return buf.getvalue()


def _truncate(text: str, length: int) -> str:
    if len(text) > length:
        return text[:length - 1] + "…"
    return text
