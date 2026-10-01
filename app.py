import io
import pandas as pd
import plotly.express as px
import streamlit as st
from whatsapp_analyzer import WhatsAppParser,ChatAnalytics
from exporters import build_workbook,build_html,build_pdf
from advanced import activity_anomalies,index_media_folder,answer_question
st.set_page_config(page_title="WhatsApp Chat Intelligence",page_icon="💬",layout="wide")
st.title("💬 WhatsApp Chat Intelligence")
st.caption("Local-first analytics for exported WhatsApp conversations. Processing stays in this app session until you download a report.")
u=st.file_uploader("Upload WhatsApp .txt export or ZIP",type=["txt","zip"])
if not u: st.info("Upload a chat export to explore activity, participants, conversations, response patterns, text, topics, links and exports."); st.stop()
try:
    raw=u.getvalue()
    if u.name.lower().endswith(".zip"):
        import zipfile
        z=zipfile.ZipFile(io.BytesIO(raw)); names=[n for n in z.namelist() if n.lower().endswith(".txt")]
        if not names: raise ValueError("ZIP does not contain a .txt chat export.")
        raw=z.read(names[0]).decode("utf-8-sig",errors="replace")
    else: raw=raw.decode("utf-8-sig",errors="replace")
    parsed=WhatsAppParser().parse_text(raw)
except Exception as e: st.error(str(e)); st.stop()
base=ChatAnalytics(parsed.dataframe)
with st.sidebar:
    st.header("Filters")
    authors=st.multiselect("Participants",base.participants,default=base.participants)
    lo,hi=base.df.DateTime.min().date(),base.df.DateTime.max().date()
    dr=st.date_input("Date range",(lo,hi),min_value=lo,max_value=hi)
    st.caption(parsed.detected_format)
    for w in parsed.warnings: st.warning(w)
view=base.df[base.df.Author.isin(authors)]
if isinstance(dr,tuple) and len(dr)==2: view=view[view.DateTime.dt.date.between(dr[0],dr[1])]
a=ChatAnalytics(view); o=a.overview()
for c,(label,key) in zip(st.columns(6),[("Messages","messages"),("Words","words"),("Participants","participants"),("Days","duration_days"),("Questions","questions"),("Links","links")]): c.metric(label,f"{o[key]:,}")
t=st.tabs(["Overview","Participants","Conversations","Text & NLP","Search","Network","Advanced","Export"])
with t[0]:
    st.plotly_chart(px.line(a.daily_activity(),x="Date_Only",y="Messages",title="Messages over time"),use_container_width=True)
    c1,c2=st.columns(2); c1.plotly_chart(px.bar(a.hourly_activity(),title="Messages by hour"),use_container_width=True); c2.plotly_chart(px.bar(a.weekday_activity(),title="Messages by weekday"),use_container_width=True)
    st.plotly_chart(px.imshow(a.heatmap(),aspect="auto",title="Activity heatmap"),use_container_width=True); st.dataframe(pd.DataFrame(a.milestones(),columns=["Milestone","When","Detail"]),use_container_width=True,hide_index=True)
with t[1]:
    p=a.participant_stats(); st.dataframe(p,use_container_width=True); st.plotly_chart(px.bar(p.reset_index(),x="Author",y="Messages",title="Messages by participant"),use_container_width=True)
with t[2]:
    gap=st.slider("New session after minutes",15,360,60); st.dataframe(a.sessions(gap).sort_values("Messages",ascending=False),use_container_width=True,hide_index=True); st.subheader("Response time"); st.dataframe(a.response_summary(),use_container_width=True); st.write("Streaks"); st.json(a.streaks())
with t[3]:
    st.plotly_chart(px.bar(a.top_words().sort_values(),orientation="h",title="Top words"),use_container_width=True); st.dataframe(a.top_phrases(),use_container_width=True,hide_index=True)
    c1,c2=st.columns(2); c1.plotly_chart(px.histogram(a.sentiment(),x="Label",title="Lexical sentiment estimate"),use_container_width=True); c2.dataframe(a.language().Language.value_counts().rename_axis("Language").reset_index(name="Messages"),use_container_width=True,hide_index=True)
    st.subheader("Topics"); [st.write(f"**Topic {x['topic']}** — {', '.join(x['terms'])}") for x in a.topics()]
with t[4]:
    q=st.text_input("Search text"); regex=st.checkbox("Regex"); typ=st.selectbox("Message type",[None,"text","image","video","audio","document","sticker","link","system"],format_func=lambda x:"Any" if x is None else x)
    try: r=a.search(q,regex=regex,message_type=typ)
    except ValueError as e: st.error(str(e)); r=a.df.iloc[0:0]
    st.dataframe(r[["DateTime","Author","Message","Message_Type"]],use_container_width=True,hide_index=True)
with t[5]: st.plotly_chart(px.imshow(a.interaction_matrix(),text_auto=True,aspect="auto",title="Participant interaction matrix"),use_container_width=True)
with t[6]:
    st.download_button("Excel",build_workbook(a),"whatsapp_analysis.xlsx"); st.download_button("HTML",build_html(a),"whatsapp_report.html"); st.download_button("PDF",build_pdf(a),"whatsapp_report.pdf"); st.download_button("CSV",a.df.to_csv(index=False),"messages.csv")
