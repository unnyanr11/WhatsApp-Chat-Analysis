"""Backward-compatible entry point for the WhatsApp Chat Intelligence engine.

New applications should import from whatsapp_analyzer.py. This module keeps
the original helper names available for notebooks and older scripts.
"""
from whatsapp_analyzer import WhatsAppParser, ChatAnalytics, analyze_file

WhatsAppChatProcessor = WhatsAppParser
WhatsAppChatAnalysis = ChatAnalytics

def get_chat_insights(df):
    a=ChatAnalytics(df); o=a.overview(); insights=[]
    if not df.empty:
        p=a.participant_stats()
        if not p.empty: insights.append(f"Most talkative: {p.index[0]} ({int(p.iloc[0].Messages):,} messages)")
        row=df.loc[df.Message_Length.idxmax()]
        insights.append(f"Longest message: {int(row.Message_Length):,} characters by {row.Author}")
        daily=df.groupby("Date_Only").size()
        if not daily.empty: insights.append(f"Most active day: {daily.idxmax()} ({int(daily.max()):,} messages)")
        if df.Emoji_Count.sum(): insights.append(f"Most emojis: {df.groupby('Author').Emoji_Count.sum().idxmax()}")
    return insights

def search_messages(df,keyword,author=None):
    return ChatAnalytics(df).search(keyword,author=author)

def get_author_stats(df,author_name):
    return df[df.Author.eq(author_name)].copy()

def analyze_whatsapp_chat(file_path):
    parsed,a=analyze_file(file_path)
    return {"dataframe":parsed.dataframe,"processor":WhatsAppParser(),"analyzer":a,"warnings":parsed.warnings}
