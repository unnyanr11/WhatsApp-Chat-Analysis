"""Optional advanced analytics: anomaly detection, media indexing and natural-language shortcuts."""
from pathlib import Path
import re
import pandas as pd

def activity_anomalies(analytics, z_threshold=2.5):
    d=analytics.daily_activity().copy()
    if d.empty:return d
    mean=d.Messages.mean(); std=d.Messages.std(ddof=0)
    d["Z_Score"]=0 if std==0 else (d.Messages-mean)/std
    d["Anomaly"]=d.Z_Score.abs()>=z_threshold
    return d.sort_values("Z_Score",ascending=False)

def index_media_folder(folder):
    root=Path(folder); rows=[]
    for p in root.rglob("*"):
        if p.is_file():
            ext=p.suffix.lower()
            kind=("image" if ext in {".jpg",".jpeg",".png",".webp",".gif",".heic"} else
                  "video" if ext in {".mp4",".mov",".avi",".mkv"} else
                  "audio" if ext in {".mp3",".m4a",".opus",".aac",".wav"} else
                  "document" if ext in {".pdf",".doc",".docx",".xls",".xlsx",".ppt",".pptx",".txt"} else "other")
            rows.append({"Path":str(p),"Filename":p.name,"Type":kind,"Extension":ext,"Bytes":p.stat().st_size})
    return pd.DataFrame(rows)

def answer_question(analytics, question):
    q=question.lower().strip()
    o=analytics.overview(); p=analytics.participant_stats()
    if "most" in q and ("message" in q or "talk" in q) and not p.empty:
        return f"{p.index[0]} has the most messages: {int(p.iloc[0].Messages):,}."
    if "busiest" in q and ("day" in q or "date" in q):
        d=analytics.daily_activity()
        if not d.empty:
            r=d.loc[d.Messages.idxmax()]; return f"The busiest day was {r.Date_Only} with {int(r.Messages):,} messages."
    if "average" in q and "response" in q:
        r=analytics.response_summary()
        return "Average response time by participant:\n"+r[["Avg_Minutes"]].to_string()
    if "longest" in q and "message" in q:
        r=analytics.df.loc[analytics.df.Message_Length.idxmax()]
        return f"The longest message was {int(r.Message_Length):,} characters, sent by {r.Author}."
    if "how many" in q and "participant" in q:
        return f"There are {o['participants']} participants."
    if "how many" in q and "message" in q:
        return f"There are {o['messages']:,} messages."
    if "emoji" in q:
        return f"The chat contains {o['emoji_count']:,} detected emojis."
    if "link" in q:
        return f"The chat contains {o['links']:,} detected URLs."
    return "I can answer built-in questions about message counts, participants, busiest days, response time, longest messages, emojis and links. Use Search for exact message retrieval."
