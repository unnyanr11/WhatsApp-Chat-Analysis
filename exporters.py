import io,html
import pandas as pd
def build_workbook(a):
    b=io.BytesIO()
    with pd.ExcelWriter(b,engine="openpyxl") as w:
        a.df.to_excel(w,index=False,sheet_name="Messages"); a.participant_stats().to_excel(w,sheet_name="Participants"); a.daily_activity().to_excel(w,index=False,sheet_name="Daily Activity"); a.response_summary().to_excel(w,sheet_name="Responses"); a.sessions().to_excel(w,index=False,sheet_name="Sessions"); a.links().to_excel(w,index=False,sheet_name="Links"); a.interaction_matrix().to_excel(w,sheet_name="Interactions")
    return b.getvalue()
def build_html(a):
    cards="".join(f"<div class='card'><b>{html.escape(str(k).replace('_',' ').title())}</b><strong>{html.escape(str(v))}</strong></div>" for k,v in a.overview().items() if k not in {"first_message","last_message"})
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>WhatsApp Chat Intelligence Report</title><style>body{{font-family:Arial;max-width:1200px;margin:40px auto;padding:0 20px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}}.card{{border:1px solid #ddd;border-radius:12px;padding:16px}}.card strong{{display:block;font-size:24px;margin-top:8px}}table{{border-collapse:collapse;width:100%}}th,td{{border-bottom:1px solid #ddd;padding:8px;text-align:left}}</style></head><body><h1>WhatsApp Chat Intelligence Report</h1><p>Generated locally from the exported chat.</p><div class="grid">{cards}</div><h2>Participants</h2>{a.participant_stats().to_html()}</body></html>"""
def build_pdf(a):
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    b=io.BytesIO(); doc=SimpleDocTemplate(b,pagesize=A4); styles=getSampleStyleSheet(); story=[Paragraph("WhatsApp Chat Intelligence Report",styles["Title"]),Spacer(1,12)]
    for k,v in a.overview().items(): story.append(Paragraph(f"<b>{k.replace('_',' ').title()}:</b> {v}",styles["BodyText"]))
    p=a.participant_stats().reset_index(); t=Table([list(p.columns)]+p.astype(str).values.tolist(),repeatRows=1); t.setStyle(TableStyle([("GRID",(0,0),(-1,-1),.25,colors.grey),("BACKGROUND",(0,0),(-1,0),colors.lightgrey)])); story += [Spacer(1,12),Paragraph("Participants",styles["Heading2"]),t]; doc.build(story); return b.getvalue()
