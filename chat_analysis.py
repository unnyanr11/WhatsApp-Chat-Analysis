
# Simple WhatsApp Chat Analysis - No Sentiment Analysis
# Complete code for Google Colab with basic chat statistics and visualizations

# Install required packages


# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from datetime import datetime
import warnings
from collections import Counter
warnings.filterwarnings('ignore')

# Text processing
from wordcloud import WordCloud

# ============================================================================
# 1. DATA PREPROCESSING
# ============================================================================

class WhatsAppChatProcessor:
    def __init__(self):
        self.df = None

    def detect_and_parse_format(self, raw_data):
        """Detect and parse different WhatsApp chat formats"""

        # Format 1: [DD/MM/YY, HH:MM:SS] Contact Name: Message
        pattern1 = r'\[(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2}:\d{2})\]\s([^:]+):\s(.+)'

        # Format 2: DD/MM/YY, HH:MM:SS - Contact Name: Message
        pattern2 = r'(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2}:\d{2})\s-\s([^:]+):\s(.+)'

        # Format 3: DD/MM/YYYY, HH:MM - Contact Name: Message
        pattern3 = r'(\d{1,2}/\d{1,2}/\d{4}),\s(\d{1,2}:\d{2})\s-\s([^:]+):\s(.+)'

        # Format 4: MM/DD/YY, HH:MM:SS AM/PM - Contact Name: Message
        pattern4 = r'(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2}:\d{2}\s(?:AM|PM))\s-\s([^:]+):\s(.+)'

        # Format 5: YYYY-MM-DD HH:MM:SS - Contact Name: Message
        pattern5 = r'(\d{4}-\d{2}-\d{2})\s(\d{2}:\d{2}:\d{2})\s-\s([^:]+):\s(.+)'

        patterns = [pattern1, pattern2, pattern3, pattern4, pattern5]

        for i, pattern in enumerate(patterns, 1):
            matches = re.findall(pattern, raw_data, re.MULTILINE)
            if matches:
                print(f"✅ Detected format {i}: Found {len(matches)} messages")
                return matches, i

        # If no pattern matches, try a more flexible approach
        print("⚠️ Standard patterns failed. Trying flexible parsing...")
        return self.flexible_parse(raw_data)

    def flexible_parse(self, raw_data):
        """Flexible parsing for unusual formats"""
        lines = raw_data.split('\n')
        messages = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Look for any date-like pattern followed by name and message
            flexible_pattern = r'(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})[,\s]*(\d{1,2}:\d{2}(?::\d{2})?(?:\s?(?:AM|PM))?)[,\s\-\[\]]*([^:]+):\s*(.+)'
            match = re.match(flexible_pattern, line)

            if match:
                date, time, author, message = match.groups()
                messages.append((date, time, author.strip(), message.strip()))

        if messages:
            print(f"✅ Flexible parsing: Found {len(messages)} messages")
            return messages, 0
        else:
            return None, None

    def show_sample_format(self, file_path, lines_to_show=10):
        """Show sample lines from the chat file to help debug format issues"""
        print("=== SAMPLE CHAT FORMAT ===")
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                for i, line in enumerate(file):
                    if i >= lines_to_show:
                        break
                    print(f"Line {i+1}: {repr(line.strip())}")
        except Exception as e:
            print(f"Error reading file: {e}")

    def load_chat(self, file_path):
        """Load and parse WhatsApp chat file with enhanced format detection"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                raw_data = file.read()
        except FileNotFoundError:
            print(f"❌ File not found: {file_path}")
            print("Make sure you've uploaded the file to Colab and the path is correct.")
            return None
        except Exception as e:
            print(f"❌ Error reading file: {e}")
            return None

        if not raw_data.strip():
            print("❌ File is empty!")
            return None

        # Show sample format for debugging
        self.show_sample_format(file_path)

        # Try to parse with different formats
        matches, format_type = self.detect_and_parse_format(raw_data)

        if not matches:
            print("❌ Could not parse any messages from the file.")
            print("\n💡 TROUBLESHOOTING TIPS:")
            print("1. Make sure you exported the chat as a .txt file (not .zip)")
            print("2. Export 'Without Media' option")
            print("3. Check the sample format above - it should contain date, time, author, and message")
            print("4. Try exporting the chat again if the format looks incorrect")
            return None

        # Create DataFrame
        df = pd.DataFrame(matches, columns=['Date', 'Time', 'Author', 'Message'])

        # Clean author names (remove extra spaces, characters)
        df['Author'] = df['Author'].str.strip()
        df['Author'] = df['Author'].str.replace(r'[\[\]]', '', regex=True)

        # Parse datetime based on format
        try:
            if format_type in [1, 2]:
                df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], errors='coerce')
            elif format_type == 3:
                df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], errors='coerce')
            elif format_type == 4:
                df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], errors='coerce')
            elif format_type == 5:
                df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], errors='coerce')
            else:  # Flexible format
                df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], errors='coerce')
        except Exception as e:
            print(f"⚠️ Warning: Some dates couldn't be parsed: {e}")

        # Clean the data
        df = df.dropna(subset=['DateTime'])
        df = df[df['Message'].str.len() > 0]  # Remove empty messages

        # Remove system messages
        system_messages = [
            'Messages and calls are end-to-end encrypted',
            'This message was deleted',
            'You deleted this message',
            '<Media omitted>',
            'Messages to this chat and calls are now secured',
            'image omitted',
            'video omitted',
            'audio omitted',
            'document omitted',
            'Contact card omitted'
        ]

        for sys_msg in system_messages:
            df = df[~df['Message'].str.contains(sys_msg, case=False, na=False)]

        # Remove very short messages (likely system messages)
        df = df[df['Message'].str.len() > 2]

        self.df = df.reset_index(drop=True)
        print(f"✅ Successfully loaded {len(self.df)} messages from {len(self.df['Author'].unique())} participants")
        print(f"📅 Date range: {self.df['DateTime'].min()} to {self.df['DateTime'].max()}")
        print(f"👥 Participants: {', '.join(self.df['Author'].unique())}")
        return self.df

    def add_features(self):
        """Add various features for analysis"""
        if self.df is None:
            print("Please load data first")
            return

        # Basic features
        self.df['Message_Length'] = self.df['Message'].str.len()
        self.df['Word_Count'] = self.df['Message'].str.split().str.len()
        self.df['Hour'] = self.df['DateTime'].dt.hour
        self.df['Day'] = self.df['DateTime'].dt.day_name()
        self.df['Month'] = self.df['DateTime'].dt.month_name()
        self.df['Year'] = self.df['DateTime'].dt.year
        self.df['Date_Only'] = self.df['DateTime'].dt.date

        # Emoji count
        emoji_pattern = re.compile("["
                                 u"\U0001F600-\U0001F64F"  # emoticons
                                 u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                 u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                 u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                 "]+", flags=re.UNICODE)

        self.df['Emoji_Count'] = self.df['Message'].apply(lambda x: len(emoji_pattern.findall(x)))

        # URL count
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        self.df['URL_Count'] = self.df['Message'].str.count(url_pattern)

        # Message type classification
        self.df['Is_Question'] = self.df['Message'].str.contains(r'\?', na=False).astype(int)
        self.df['Is_Media'] = self.df['Message'].str.contains('<Media omitted>', na=False).astype(int)

        # Check for common patterns
        self.df['Has_Emoji'] = (self.df['Emoji_Count'] > 0).astype(int)
        self.df['Has_URL'] = (self.df['URL_Count'] > 0).astype(int)

        print("✅ Features added successfully!")

        return self.df

# ============================================================================
# 2. CHAT ANALYSIS AND STATISTICS
# ============================================================================

class WhatsAppChatAnalysis:
    def __init__(self, df):
        self.df = df

    def basic_stats(self):
        """Display comprehensive basic statistics"""
        print("=" * 60)
        print("📊 WHATSAPP CHAT ANALYSIS REPORT")
        print("=" * 60)

        # Basic metrics
        total_messages = len(self.df)
        total_words = self.df['Word_Count'].sum()
        total_characters = self.df['Message_Length'].sum()
        date_range = (self.df['DateTime'].max() - self.df['DateTime'].min()).days

        print(f"\n📈 OVERVIEW:")
        print(f"Total Messages: {total_messages:,}")
        print(f"Total Words: {total_words:,}")
        print(f"Total Characters: {total_characters:,}")
        print(f"Chat Duration: {date_range} days")
        print(f"Date Range: {self.df['DateTime'].min().strftime('%Y-%m-%d')} to {self.df['DateTime'].max().strftime('%Y-%m-%d')}")
        print(f"Average Messages per Day: {total_messages/max(date_range, 1):.1f}")

        # Author statistics
        print(f"\n👥 PARTICIPANTS ({len(self.df['Author'].unique())}):")
        author_stats = self.df['Author'].value_counts()
        for author, count in author_stats.items():
            percentage = (count / total_messages) * 100
            avg_words = self.df[self.df['Author'] == author]['Word_Count'].mean()
            avg_length = self.df[self.df['Author'] == author]['Message_Length'].mean()
            print(f"  {author}:")
            print(f"    Messages: {count:,} ({percentage:.1f}%)")
            print(f"    Avg Words per Message: {avg_words:.1f}")
            print(f"    Avg Characters per Message: {avg_length:.1f}")

        # Message characteristics
        print(f"\n📝 MESSAGE CHARACTERISTICS:")
        print(f"Average Message Length: {self.df['Message_Length'].mean():.1f} characters")
        print(f"Average Words per Message: {self.df['Word_Count'].mean():.1f}")
        print(f"Longest Message: {self.df['Message_Length'].max()} characters")
        print(f"Messages with Emojis: {self.df['Has_Emoji'].sum():,} ({(self.df['Has_Emoji'].sum()/total_messages)*100:.1f}%)")
        print(f"Messages with URLs: {self.df['Has_URL'].sum():,} ({(self.df['Has_URL'].sum()/total_messages)*100:.1f}%)")
        print(f"Questions Asked: {self.df['Is_Question'].sum():,} ({(self.df['Is_Question'].sum()/total_messages)*100:.1f}%)")

        # Activity patterns
        print(f"\n⏰ ACTIVITY PATTERNS:")

        # Most active hour
        hourly_counts = self.df['Hour'].value_counts().sort_index()
        most_active_hour = hourly_counts.idxmax()
        print(f"Most Active Hour: {most_active_hour}:00 ({hourly_counts[most_active_hour]} messages)")

        # Most active day
        daily_counts = self.df['Day'].value_counts()
        most_active_day = daily_counts.idxmax()
        print(f"Most Active Day: {most_active_day} ({daily_counts[most_active_day]} messages)")

        # Most active month
        monthly_counts = self.df['Month'].value_counts()
        most_active_month = monthly_counts.idxmax()
        print(f"Most Active Month: {most_active_month} ({monthly_counts[most_active_month]} messages)")

        # Busiest chat day
        daily_message_counts = self.df['Date_Only'].value_counts()
        busiest_day = daily_message_counts.idxmax()
        print(f"Busiest Chat Day: {busiest_day} ({daily_message_counts[busiest_day]} messages)")

        print("=" * 60)

    def plot_activity_patterns(self):
        """Plot various activity patterns"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. Messages over time
        daily_messages = self.df.groupby('Date_Only').size()
        daily_messages.plot(kind='line', ax=axes[0,0], color='blue')
        axes[0,0].set_title('Messages Over Time')
        axes[0,0].set_xlabel('Date')
        axes[0,0].set_ylabel('Number of Messages')
        axes[0,0].tick_params(axis='x', rotation=45)

        # 2. Messages by hour
        hourly_messages = self.df['Hour'].value_counts().sort_index()
        hourly_messages.plot(kind='bar', ax=axes[0,1], color='green')
        axes[0,1].set_title('Messages by Hour of Day')
        axes[0,1].set_xlabel('Hour')
        axes[0,1].set_ylabel('Number of Messages')

        # 3. Messages by day of week
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_messages = self.df['Day'].value_counts().reindex(day_order, fill_value=0)
        day_messages.plot(kind='bar', ax=axes[0,2], color='orange')
        axes[0,2].set_title('Messages by Day of Week')
        axes[0,2].set_xlabel('Day')
        axes[0,2].set_ylabel('Number of Messages')
        axes[0,2].tick_params(axis='x', rotation=45)

        # 4. Messages by author
        author_messages = self.df['Author'].value_counts()
        author_messages.plot(kind='bar', ax=axes[1,0], color='red')
        axes[1,0].set_title('Messages by Author')
        axes[1,0].set_xlabel('Author')
        axes[1,0].set_ylabel('Number of Messages')
        axes[1,0].tick_params(axis='x', rotation=45)

        # 5. Message length distribution
        self.df['Message_Length'].hist(bins=30, ax=axes[1,1], color='purple')
        axes[1,1].set_title('Message Length Distribution')
        axes[1,1].set_xlabel('Message Length (characters)')
        axes[1,1].set_ylabel('Frequency')

        # 6. Word count distribution
        self.df['Word_Count'].hist(bins=30, ax=axes[1,2], color='brown')
        axes[1,2].set_title('Word Count Distribution')
        axes[1,2].set_xlabel('Words per Message')
        axes[1,2].set_ylabel('Frequency')

        plt.tight_layout()
        plt.show()

    def plot_author_comparison(self):
        """Compare authors across different metrics"""
        if len(self.df['Author'].unique()) < 2:
            print("Need at least 2 authors for comparison")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. Message count by author
        author_counts = self.df['Author'].value_counts()
        author_counts.plot(kind='pie', ax=axes[0,0], autopct='%1.1f%%')
        axes[0,0].set_title('Message Distribution by Author')

        # 2. Average message length by author
        avg_length = self.df.groupby('Author')['Message_Length'].mean()
        avg_length.plot(kind='bar', ax=axes[0,1], color='skyblue')
        axes[0,1].set_title('Average Message Length by Author')
        axes[0,1].set_ylabel('Characters')
        axes[0,1].tick_params(axis='x', rotation=45)

        # 3. Average words per message by author
        avg_words = self.df.groupby('Author')['Word_Count'].mean()
        avg_words.plot(kind='bar', ax=axes[1,0], color='lightgreen')
        axes[1,0].set_title('Average Words per Message by Author')
        axes[1,0].set_ylabel('Words')
        axes[1,0].tick_params(axis='x', rotation=45)

        # 4. Emoji usage by author
        emoji_usage = self.df.groupby('Author')['Emoji_Count'].sum()
        emoji_usage.plot(kind='bar', ax=axes[1,1], color='gold')
        axes[1,1].set_title('Total Emoji Usage by Author')
        axes[1,1].set_ylabel('Number of Emojis')
        axes[1,1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()

    def plot_time_heatmap(self):
        """Create a heatmap of activity by hour and day"""
        # Create hour vs day matrix
        activity_matrix = self.df.groupby(['Day', 'Hour']).size().unstack(fill_value=0)

        # Reorder days
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        activity_matrix = activity_matrix.reindex(day_order, fill_value=0)

        plt.figure(figsize=(15, 8))
        sns.heatmap(activity_matrix, annot=True, fmt='d', cmap='YlOrRd', cbar_kws={'label': 'Number of Messages'})
        plt.title('Chat Activity Heatmap (Day vs Hour)')
        plt.xlabel('Hour of Day')
        plt.ylabel('Day of Week')
        plt.show()

    def generate_wordcloud(self):
        """Generate word cloud from all messages"""
        try:
            # Combine all messages
            all_messages = ' '.join(self.df['Message'].astype(str))

            if len(all_messages.strip()) == 0:
                print("No text available for word cloud")
                return

            # Basic stopwords (since we're not using NLTK)
            stop_words = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                'should', 'may', 'might', 'must', 'can', 'i', 'you', 'he', 'she',
                'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your',
                'his', 'her', 'its', 'our', 'their', 'this', 'that', 'these', 'those'
            }

            # Clean text
            cleaned_text = re.sub(r'[^a-zA-Z\s]', '', all_messages.lower())

            if len(cleaned_text.strip()) == 0:
                print("No valid text after cleaning for word cloud")
                return

            wordcloud = WordCloud(
                width=1200,
                height=600,
                background_color='white',
                stopwords=stop_words,
                max_words=100,
                colormap='viridis'
            ).generate(cleaned_text)

            plt.figure(figsize=(15, 8))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Most Common Words in Chat', fontsize=20, pad=20)
            plt.show()
        except Exception as e:
            print(f"Error creating word cloud: {e}")

    def author_specific_wordcloud(self):
        """Generate separate word clouds for each author"""
        authors = self.df['Author'].unique()

        if len(authors) > 4:
            print(f"Too many authors ({len(authors)}). Showing top 4 most active.")
            top_authors = self.df['Author'].value_counts().head(4).index
            authors = top_authors

        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        axes = axes.flatten()

        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can', 'i', 'you', 'he', 'she',
            'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your',
            'his', 'her', 'its', 'our', 'their', 'this', 'that', 'these', 'those'
        }

        for i, author in enumerate(authors[:4]):
            if i >= 4:
                break

            author_messages = self.df[self.df['Author'] == author]['Message']
            text = ' '.join(author_messages.astype(str))
            cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text.lower())

            if len(cleaned_text.strip()) > 0:
                try:
                    wordcloud = WordCloud(
                        width=600,
                        height=400,
                        background_color='white',
                        stopwords=stop_words,
                        max_words=50,
                        colormap='Set2'
                    ).generate(cleaned_text)

                    axes[i].imshow(wordcloud, interpolation='bilinear')
                    axes[i].set_title(f'{author}', fontsize=14)
                    axes[i].axis('off')
                except:
                    axes[i].text(0.5, 0.5, f'Not enough text\nfor {author}',
                               ha='center', va='center', transform=axes[i].transAxes)
                    axes[i].set_title(f'{author}', fontsize=14)
                    axes[i].axis('off')
            else:
                axes[i].text(0.5, 0.5, f'No valid text\nfor {author}',
                           ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f'{author}', fontsize=14)
                axes[i].axis('off')

        # Hide unused subplots
        for i in range(len(authors), 4):
            axes[i].axis('off')

        plt.tight_layout()
        plt.suptitle('Word Clouds by Author', fontsize=16, y=1.02)
        plt.show()

    def monthly_activity_trend(self):
        """Show activity trends by month"""
        monthly_data = self.df.groupby([self.df['DateTime'].dt.to_period('M')]).size()

        plt.figure(figsize=(15, 6))
        monthly_data.plot(kind='line', marker='o', linewidth=2, markersize=8)
        plt.title('Monthly Chat Activity Trend', fontsize=16)
        plt.xlabel('Month')
        plt.ylabel('Number of Messages')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def top_chatters_by_time(self):
        """Show who is most active at different times"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Morning (6-12)
        morning_data = self.df[self.df['Hour'].between(6, 11)]['Author'].value_counts()
        if len(morning_data) > 0:
            morning_data.head(5).plot(kind='bar', ax=axes[0], color='gold')
            axes[0].set_title('Most Active in Morning (6 AM - 12 PM)')
            axes[0].set_ylabel('Messages')
            axes[0].tick_params(axis='x', rotation=45)

        # Afternoon (12-18)
        afternoon_data = self.df[self.df['Hour'].between(12, 17)]['Author'].value_counts()
        if len(afternoon_data) > 0:
            afternoon_data.head(5).plot(kind='bar', ax=axes[1], color='orange')
            axes[1].set_title('Most Active in Afternoon (12 PM - 6 PM)')
            axes[1].set_ylabel('Messages')
            axes[1].tick_params(axis='x', rotation=45)

        # Night (18-24)
        night_data = self.df[self.df['Hour'].between(18, 23)]['Author'].value_counts()
        if len(night_data) > 0:
            night_data.head(5).plot(kind='bar', ax=axes[2], color='navy')
            axes[2].set_title('Most Active at Night (6 PM - 12 AM)')
            axes[2].set_ylabel('Messages')
            axes[2].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()

    def generate_detailed_report(self):
        """Generate a comprehensive text report"""
        report = []
        report.append("=" * 80)
        report.append("WHATSAPP CHAT ANALYSIS - DETAILED REPORT")
        report.append("=" * 80)

        # Basic stats
        total_messages = len(self.df)
        date_range = (self.df['DateTime'].max() - self.df['DateTime'].min()).days

        report.append(f"\nOVERVIEW:")
        report.append(f"Total Messages: {total_messages:,}")
        report.append(f"Chat Duration: {date_range} days")
        report.append(f"Average Messages per Day: {total_messages/max(date_range, 1):.1f}")

        # Author stats
        report.append(f"\nPARTICIPANT STATISTICS:")
        author_stats = self.df['Author'].value_counts()
        for author, count in author_stats.items():
            percentage = (count / total_messages) * 100
            avg_words = self.df[self.df['Author'] == author]['Word_Count'].mean()
            report.append(f"{author}: {count:,} messages ({percentage:.1f}%), avg {avg_words:.1f} words/msg")

        # Activity patterns
        report.append(f"\nACTIVITY PATTERNS:")
        hourly_counts = self.df['Hour'].value_counts().sort_index()
        most_active_hour = hourly_counts.idxmax()
        report.append(f"Most Active Hour: {most_active_hour}:00 ({hourly_counts[most_active_hour]} messages)")

        daily_counts = self.df['Day'].value_counts()
        most_active_day = daily_counts.idxmax()
        report.append(f"Most Active Day: {most_active_day} ({daily_counts[most_active_day]} messages)")

        # Message characteristics
        report.append(f"\nMESSAGE CHARACTERISTICS:")
        report.append(f"Average Message Length: {self.df['Message_Length'].mean():.1f} characters")
        report.append(f"Messages with Emojis: {self.df['Has_Emoji'].sum():,} ({(self.df['Has_Emoji'].sum()/total_messages)*100:.1f}%)")
        report.append(f"Questions Asked: {self.df['Is_Question'].sum():,} ({(self.df['Is_Question'].sum()/total_messages)*100:.1f}%)")

        report.append("=" * 80)

        # Print and save report
        full_report = '\n'.join(report)
        print(full_report)

        # Save to file
        with open('whatsapp_analysis_report.txt', 'w', encoding='utf-8') as f:
            f.write(full_report)
        print("\n✅ Report saved as 'whatsapp_analysis_report.txt'")

        return full_report

# ============================================================================
# 3. MAIN EXECUTION PIPELINE
# ============================================================================

def analyze_whatsapp_chat(file_path):
    """Main function to analyze WhatsApp chat"""
    print("🚀 Starting WhatsApp Chat Analysis...")

    # 1. Load and preprocess data
    print("\n📊 Step 1: Loading and preprocessing data...")
    processor = WhatsAppChatProcessor()
    df = processor.load_chat(file_path)

    if df is None:
        print("❌ Failed to load data. Please check your file format and try again.")
        return None

    df = processor.add_features()

    # 2. Perform analysis
    print("\n📈 Step 2: Analyzing chat data...")
    analyzer = WhatsAppChatAnalysis(df)

    # Display basic statistics
    analyzer.basic_stats()

    # Generate visualizations
    print("\n📊 Generating visualizations...")

    print("1. Activity Patterns...")
    analyzer.plot_activity_patterns()

    if len(df['Author'].unique()) >= 2:
        print("2. Author Comparison...")
        analyzer.plot_author_comparison()

    print("3. Time Heatmap...")
    analyzer.plot_time_heatmap()

    print("4. Overall Word Cloud...")
    analyzer.generate_wordcloud()

    if len(df['Author'].unique()) >= 2:
        print("5. Author-specific Word Clouds...")
        analyzer.author_specific_wordcloud()

    print("6. Monthly Activity Trend...")
    analyzer.monthly_activity_trend()

    print("7. Time-based Activity Analysis...")
    analyzer.top_chatters_by_time()

    # Generate detailed report
    print("\n📋 Generating detailed report...")
    analyzer.generate_detailed_report()

    print("\n✅ Analysis completed successfully!")

    return {
        'dataframe': df,
        'processor': processor,
        'analyzer': analyzer
    }

# ============================================================================
# 4. INTERACTIVE FUNCTIONS
# ============================================================================

def get_chat_insights(df):
    """Get quick insights about the chat"""
    insights = []

    # Most talkative person
    most_talkative = df['Author'].value_counts().index[0]
    most_talkative_count = df['Author'].value_counts().iloc[0]
    total_messages = len(df)

    insights.append(f"🗣️ Most talkative: {most_talkative} ({most_talkative_count} messages, {(most_talkative_count/total_messages)*100:.1f}%)")

    # Longest message
    longest_msg_idx = df['Message_Length'].idxmax()
    longest_msg_author = df.loc[longest_msg_idx, 'Author']
    longest_msg_length = df.loc[longest_msg_idx, 'Message_Length']

    insights.append(f"📝 Longest message: {longest_msg_length} characters by {longest_msg_author}")

    # Most active day
    daily_counts = df.groupby(df['DateTime'].dt.date).size()
    most_active_date = daily_counts.idxmax()
    most_active_count = daily_counts.max()

    insights.append(f"📅 Most active day: {most_active_date} ({most_active_count} messages)")

    # Emoji lover
    emoji_counts = df.groupby('Author')['Emoji_Count'].sum()
    if emoji_counts.sum() > 0:
        emoji_lover = emoji_counts.idxmax()
        emoji_count = emoji_counts.max()
        insights.append(f"😄 Emoji lover: {emoji_lover} ({emoji_count} emojis)")

    # Question asker
    question_counts = df.groupby('Author')['Is_Question'].sum()
    if question_counts.sum() > 0:
        question_asker = question_counts.idxmax()
        question_count = question_counts.max()
        insights.append(f"❓ Most curious: {question_asker} ({question_count} questions)")

    return insights

def search_messages(df, keyword, author=None):
    """Search for messages containing a keyword"""
    # Filter by keyword
    filtered_df = df[df['Message'].str.contains(keyword, case=False, na=False)]

    # Filter by author if specified
    if author:
        filtered_df = filtered_df[filtered_df['Author'] == author]

    if len(filtered_df) == 0:
        print(f"No messages found containing '{keyword}'")
        return None

    print(f"Found {len(filtered_df)} messages containing '{keyword}':")
    print("-" * 80)

    for idx, row in filtered_df.head(10).iterrows():  # Show top 10 results
        print(f"[{row['DateTime'].strftime('%Y-%m-%d %H:%M')}] {row['Author']}: {row['Message']}")

    if len(filtered_df) > 10:
        print(f"... and {len(filtered_df) - 10} more results")

    return filtered_df

def get_author_stats(df, author_name):
    """Get detailed statistics for a specific author"""
    author_df = df[df['Author'] == author_name]

    if len(author_df) == 0:
        print(f"Author '{author_name}' not found")
        return None

    print(f"📊 STATISTICS FOR {author_name.upper()}")
    print("-" * 50)
    print(f"Total Messages: {len(author_df):,}")
    print(f"Total Words: {author_df['Word_Count'].sum():,}")
    print(f"Average Message Length: {author_df['Message_Length'].mean():.1f} characters")
    print(f"Average Words per Message: {author_df['Word_Count'].mean():.1f}")
    print(f"Emojis Used: {author_df['Emoji_Count'].sum():,}")
    print(f"Questions Asked: {author_df['Is_Question'].sum():,}")
    print(f"URLs Shared: {author_df['URL_Count'].sum():,}")

    # Most active times
    most_active_hour = author_df['Hour'].value_counts().index[0]
    most_active_day = author_df['Day'].value_counts().index[0]

    print(f"Most Active Hour: {most_active_hour}:00")
    print(f"Most Active Day: {most_active_day}")

    return author_df

# ============================================================================
# 5. MAIN EXECUTION
# ============================================================================

# Update the file path to your chat file
file_path = "/content/r.txt"  # Change this to your file path

# Run the complete analysis
results = analyze_whatsapp_chat(file_path)

if results:
    df = results['dataframe']

    # Show quick insights
    print("\n🔍 QUICK INSIGHTS:")
    insights = get_chat_insights(df)
    for insight in insights:
        print(insight)

    # Example usage of interactive functions
    print("\n" + "="*60)
    print("🔧 INTERACTIVE FUNCTIONS AVAILABLE:")
    print("="*60)
    print("1. search_messages(df, 'keyword') - Search for messages")
    print("2. get_author_stats(df, 'Author Name') - Get author statistics")
    print("3. get_chat_insights(df) - Get quick insights")
    print("\nExample usage:")
    print("search_messages(df, 'hello')")
    print("get_author_stats(df, 'John')")  # Replace with actual author name

    # Show available authors
    print(f"\nAvailable authors: {', '.join(df['Author'].unique())}")

else:
    print("❌ Analysis failed. Please check your file format and path.")
    print("\n💡 TROUBLESHOOTING:")
    print("1. Make sure the file path is correct")
    print("2. Ensure the file is uploaded to Colab")
    print("3. Check that the file is a .txt export from WhatsApp")
    print("4. Make sure you exported 'Without Media'")