from whatsapp_analyzer import WhatsAppParser,ChatAnalytics
SAMPLE="""31/10/19, 9:19 pm - A: Hello
second line
31/10/19, 9:20 pm - B: https://example.com
31/10/19, 9:22 pm - A: 😂 Great!
"""
def test_multiline_and_features():
    df=WhatsAppParser(dayfirst=True).parse_text(SAMPLE).dataframe
    assert len(df)==3 and "second line" in df.loc[0,"Message"] and df.loc[1,"URL_Count"]==1 and df.loc[2,"Emoji_Count"]>=1
def test_analytics():
    a=ChatAnalytics(WhatsAppParser(dayfirst=True).parse_text(SAMPLE).dataframe)
    assert a.overview()["messages"]==3 and set(a.participants)=={"A","B"} and len(a.sessions())==1
