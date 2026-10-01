"""Privacy-first WhatsApp chat analytics engine."""
from __future__ import annotations
import re, zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import pandas as pd

MESSAGE_RE=re.compile(r"^\[?(?P<date>\d{1,4}[./-]\d{1,2}[./-]\d{1,4}),?\s+(?P<time>\d{1,2}:\d{2}(?::\d{2})?\s?(?:AM|PM|am|pm)?)\]?\s*[-–]\s+(?P<rest>.*)$")
SYSTEM_PATTERNS=("messages and calls are end-to-end encrypted","messages to this chat and calls are now secured","this message was deleted","you deleted this message","changed the subject","changed the group icon","added "," removed "," left","joined using this group's invite link")
MEDIA_PATTERNS={"image":("image omitted","<media omitted>"),"video":("video omitted",),"audio":("audio omitted",),"document":("document omitted",),"sticker":("sticker omitted",),"contact":("contact card omitted",),"location":("location:",)}

@dataclass
class ParseResult:
    dataframe: pd.DataFrame
    detected_format: str
    warnings: list[str]

class WhatsAppParser:
    def __init__(self, dayfirst: Optional[bool]=None): self.dayfirst=dayfirst
    def parse_text(self, raw: str)->ParseResult:
        rows=[]; warnings=[]; current=None
        for line in raw.replace("\ufeff","").splitlines():
            m=MESSAGE_RE.match(line.rstrip("\r"))
            if m:
                if current: rows.append(current)
                date,time,rest=m.group("date"),m.group("time"),m.group("rest")
                if ": " in rest:
                    author,message=rest.split(": ",1); current=[date,time,author.strip(),message]
                else: current=[date,time,"System",rest]
            elif current and line.strip(): current[3]+="\n"+line
        if current: rows.append(current)
        if not rows: raise ValueError("No WhatsApp messages detected. Export the chat as a .txt file.")
        df=pd.DataFrame(rows,columns=["Date","Time","Author","Message"])
        dt=pd.to_datetime(df.Date+" "+df.Time,errors="coerce",dayfirst=self.dayfirst)
        if self.dayfirst is None and dt.notna().sum()<len(df)*.8: dt=pd.to_datetime(df.Date+" "+df.Time,errors="coerce",dayfirst=True)
        df["DateTime"]=dt
        bad=int(df.DateTime.isna().sum())
        if bad: warnings.append(f"Dropped {bad} rows with unparseable dates."); df=df.dropna(subset=["DateTime"])
        df["Author"]=df.Author.fillna("System").astype(str).str.strip()
        df["Message"]=df.Message.fillna("").astype(str)
        df["Is_System"]=df.Author.eq("System")|df.Message.str.lower().apply(self._is_system)
        df["Message_Type"]=df.Message.str.lower().apply(self._message_type)
        df["Is_Deleted"]=df.Message.str.lower().str.contains("message was deleted|you deleted this message",regex=True)
        df=self._features(df); df.insert(0,"Message_ID",range(1,len(df)+1))
        return ParseResult(df.reset_index(drop=True),self._detect_format(df),warnings)
    def parse_file(self,path):
        path=Path(path)
        if path.suffix.lower()==".zip":
            with zipfile.ZipFile(path) as z:
                txt=next((n for n in z.namelist() if n.lower().endswith(".txt")),None)
                if not txt: raise ValueError("ZIP does not contain a WhatsApp .txt export.")
                return self.parse_text(z.read(txt).decode("utf-8-sig",errors="replace"))
        return self.parse_text(path.read_text(encoding="utf-8-sig",errors="replace"))
    @staticmethod
    def _detect_format(df):
        return "ISO" if df.Date.astype(str).str.match(r"\d{4}-\d{1,2}-\d{1,2}").any() else "WhatsApp export"
    @staticmethod
    def _is_system(text):
        t=str(text).strip().lower()
        return any(t.startswith(p) or p in t for p in SYSTEM_PATTERNS)
    @staticmethod
    def _message_type(text):
        t=str(text).strip().lower()
        for kind,patterns in MEDIA_PATTERNS.items():
            if any(p in t for p in patterns): return kind
        if re.search(r"https?://|www\.",t): return "link"
        if "voice call" in t: return "voice_call"
        if "video call" in t: return "video_call"
        return "system" if WhatsAppParser._is_system(t) else "text"
    @staticmethod
    def _features(df):
        emoji=re.compile(r"[\U0001F1E6-\U0001F1FF\U0001F300-\U0001FAFF\u2600-\u27BF]",re.UNICODE)
        url=re.compile(r"(?:https?://|www\.)\S+",re.I)
        df["Message_Length"]=df.Message.str.len(); df["Word_Count"]=df.Message.str.findall(r"\b\w+\b",flags=re.UNICODE).str.len()
        df["Emoji_Count"]=df.Message.apply(lambda x:len(emoji.findall(x))); df["URL_Count"]=df.Message.apply(lambda x:len(url.findall(x)))
        df["Is_Question"]=df.Message.str.contains(r"\?").astype(int); df["Is_Exclamation"]=df.Message.str.contains("!").astype(int)
        df["Has_Emoji"]=(df.Emoji_Count>0).astype(int); df["Has_URL"]=(df.URL_Count>0).astype(int)
        df["Date_Only"]=df.DateTime.dt.date; df["Year"]=df.DateTime.dt.year; df["Month"]=df.DateTime.dt.to_period("M").astype(str)
        df["Day"]=df.DateTime.dt.day_name(); df["Hour"]=df.DateTime.dt.hour; df["Is_Weekend"]=df.DateTime.dt.dayofweek>=5
        df["Domain"]=df.Message.str.extract(r"(?:https?://|www\.)(?:www\.)?([^/\s]+)",expand=False).fillna("")
        return df

class ChatAnalytics:
    def __init__(self,df): self.df=df.copy().sort_values("DateTime").reset_index(drop=True); self._sessions={}
    @property
    def participants(self): return sorted(self.df.loc[~self.df.Is_System,"Author"].unique().tolist())
    def overview(self):
        d=self.df; duration=(d.DateTime.max()-d.DateTime.min()).total_seconds()/86400 if len(d) else 0
        return {"messages":len(d),"words":int(d.Word_Count.sum()),"characters":int(d.Message_Length.sum()),"participants":len(self.participants),"duration_days":round(duration,1),"avg_messages_per_day":round(len(d)/max(duration,1),2),"avg_words_per_message":round(d.Word_Count.mean(),2),"media_messages":int((d.Message_Type!="text").sum()),"questions":int(d.Is_Question.sum()),"emoji_count":int(d.Emoji_Count.sum()),"links":int(d.URL_Count.sum()),"first_message":d.DateTime.min(),"last_message":d.DateTime.max()}
    def participant_stats(self):
        d=self.df[~self.df.Is_System]; g=d.groupby("Author")
        o=g.agg(Messages=("Message_ID","count"),Words=("Word_Count","sum"),Avg_Words=("Word_Count","mean"),Avg_Length=("Message_Length","mean"),Emojis=("Emoji_Count","sum"),Questions=("Is_Question","sum"),Links=("URL_Count","sum"),Media=("Message_Type",lambda s:(s!="text").sum()))
        o["Share_%"]=o.Messages/max(len(d),1)*100; o["Peak_Hour"]=g.Hour.agg(lambda s:int(s.mode().iloc[0]) if len(s.mode()) else 0)
        return o.sort_values("Messages",ascending=False).round(2)
    def daily_activity(self): return self.df.groupby("Date_Only").agg(Messages=("Message_ID","count"),Words=("Word_Count","sum")).reset_index()
    def hourly_activity(self): return self.df.groupby("Hour").size().reindex(range(24),fill_value=0)
    def weekday_activity(self):
        order=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]; return self.df.groupby("Day").size().reindex(order,fill_value=0)
    def heatmap(self):
        order=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]; return self.df.pivot_table(index="Day",columns="Hour",values="Message_ID",aggfunc="count",fill_value=0).reindex(order,fill_value=0)
    def sessions(self,gap_minutes=60):
        if gap_minutes in self._sessions: return self._sessions[gap_minutes]
        d=self.df.copy(); gap=d.DateTime.diff().dt.total_seconds().div(60).fillna(0); d["Session_ID"]=(gap>gap_minutes).cumsum()+1
        s=d.groupby("Session_ID").agg(Start=("DateTime","min"),End=("DateTime","max"),Messages=("Message_ID","count"),Participants=("Author","nunique")); s["Duration_Minutes"]=(s.End-s.Start).dt.total_seconds().div(60).round(1); s["Initiator"]=d.groupby("Session_ID").first().Author
        self._sessions[gap_minutes]=s.reset_index(); return self._sessions[gap_minutes]
    def response_stats(self,gap_minutes=360):
        d=self.df[~self.df.Is_System].copy(); delta=d.DateTime.diff().dt.total_seconds().div(60); changed=d.Author.ne(d.Author.shift()); valid=changed&(delta>=0)&(delta<=gap_minutes)
        r=d.loc[valid,["Message_ID","Author","DateTime"]].copy(); r["Response_Minutes"]=delta.loc[valid].round(2).values; return r
    def response_summary(self,gap_minutes=360):
        r=self.response_stats(gap_minutes)
        if r.empty:return pd.DataFrame(columns=["Author","Responses","Avg_Minutes","Median_Minutes","Fastest_Minutes"])
        return r.groupby("Author").agg(Responses=("Message_ID","count"),Avg_Minutes=("Response_Minutes","mean"),Median_Minutes=("Response_Minutes","median"),Fastest_Minutes=("Response_Minutes","min")).round(2)
    def streaks(self):
        dates=pd.Series(sorted(pd.unique(self.df.Date_Only)))
        if dates.empty:return {"longest_days":0,"active_days":0,"longest_silence_days":0}
        dif=pd.to_datetime(dates).diff().dt.days.fillna(1); runs=dates.groupby((dif>1).cumsum()).size()
        return {"longest_days":int(runs.max()),"active_days":len(dates),"longest_silence_days":int(max(dif.max()-1,0))}
    def top_words(self,author=None,n=30):
        d=self.df[~self.df.Is_System]
        if author:d=d[d.Author.eq(author)]
        words=re.findall(r"(?u)\b[\w’'-]{2,}\b"," ".join(d.Message.astype(str)).lower())
        stop=set("the and for that this with you your are was were have has had but not what how can just from they them our his her she he its to of in on at is it i me we a an or as be do did so if my".split())
        return pd.Series(Counter(w for w in words if w not in stop)).head(n)
    def top_phrases(self,n=20):
        big,tri=Counter(),Counter()
        for t in self.df.loc[~self.df.Is_System,"Message"].astype(str).str.lower():
            ts=re.findall(r"(?u)\b\w+\b",t); big.update(zip(ts,ts[1:])); tri.update(zip(ts,ts[1:],ts[2:]))
        return pd.DataFrame({"bigram":[" ".join(x) for x,_ in big.most_common(n)],"bigram_count":[c for _,c in big.most_common(n)],"trigram":[" ".join(x) for x,_ in tri.most_common(n)],"trigram_count":[c for _,c in tri.most_common(n)]})
    def links(self): return self.df[self.df.Has_URL][["DateTime","Author","Domain","Message"]].sort_values("DateTime",ascending=False)
    def interaction_matrix(self):
        d=self.df[~self.df.Is_System]; prev=d.Author.shift(); mask=prev.notna()&prev.ne(d.Author); pairs=pd.DataFrame({"Previous":prev[mask].values,"Author":d.loc[mask,"Author"].values}); return pd.crosstab(pairs.Previous,pairs.Author).reindex(index=self.participants,columns=self.participants,fill_value=0)
    def milestones(self):
        d=self.df
        if d.empty:return []
        daily=d.groupby("Date_Only").size(); rows=[("First message",d.iloc[0].DateTime,d.iloc[0].Author),("Last message",d.iloc[-1].DateTime,d.iloc[-1].Author),("Busiest day",pd.Timestamp(daily.idxmax()),f"{int(daily.max()):,} messages"),("Longest message",d.loc[d.Message_Length.idxmax(),"DateTime"],d.loc[d.Message_Length.idxmax(),"Author"])]
        if d.Emoji_Count.sum(): rows.append(("First emoji",d.loc[d.Emoji_Count.gt(0)].iloc[0].DateTime,""))
        return rows
    def search(self,query,author=None,start=None,end=None,regex=False,message_type=None):
        d=self.df.copy()
        if query:
            d=d[d.Message.str.contains(query,case=False,regex=regex,na=False)]
        if author:d=d[d.Author.eq(author)]
        if start:d=d[d.DateTime.ge(pd.Timestamp(start))]
        if end:d=d[d.DateTime.le(pd.Timestamp(end))]
        if message_type:d=d[d.Message_Type.eq(message_type)]
        return d.sort_values("DateTime",ascending=False)
    def sentiment(self):
        pos=set("good great amazing awesome love loved happy haha nice thanks thank beautiful best excellent wow".split()); neg=set("bad hate hated sad angry awful terrible worst stupid sorry upset problem".split())
        def score(t):
            ws=re.findall(r"\b\w+\b",str(t).lower()); return sum(w in pos for w in ws)-sum(w in neg for w in ws)
        s=self.df.Message.map(score); return pd.DataFrame({"DateTime":self.df.DateTime,"Author":self.df.Author,"Score":s,"Label":s.map(lambda x:"positive" if x>0 else "negative" if x<0 else "neutral")})
    def language(self):
        try: from langdetect import detect
        except ImportError:return pd.DataFrame({"Message_ID":self.df.Message_ID,"Language":["unavailable"]*len(self.df)})
        def f(t):
            try:return detect(str(t))
            except:return "unknown"
        return pd.DataFrame({"Message_ID":self.df.Message_ID,"Language":self.df.Message.map(f)})
    def topics(self,n_topics=6,n_terms=8):
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.decomposition import NMF
        except ImportError:return []
        texts=self.df.loc[~self.df.Is_System,"Message"].astype(str)
        if len(texts)<n_topics:return []
        vec=TfidfVectorizer(stop_words="english",ngram_range=(1,2),min_df=2,max_features=5000); X=vec.fit_transform(texts)
        if X.shape[1]==0:return []
        model=NMF(n_components=min(n_topics,X.shape[1],len(texts)),random_state=42,init="nndsvda",max_iter=300).fit(X); terms=vec.get_feature_names_out()
        return [{"topic":i+1,"terms":[terms[j] for j in comp.argsort()[-n_terms:][::-1]]} for i,comp in enumerate(model.components_)]

def analyze_file(path,dayfirst=None):
    parsed=WhatsAppParser(dayfirst).parse_file(path); return parsed,ChatAnalytics(parsed.dataframe)
