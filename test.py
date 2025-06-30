import streamlit as st
import pandas as pd
from transformers import pipeline
import asyncio
import feedparser
import requests
import re
from collections import Counter
from urllib.parse import quote
from bs4 import BeautifulSoup
import yfinance as yf
import time
import plotly.express as px
import plotly.graph_objects as go
import threading
import sys
import platform

# Check if Playwright is available
try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    st.error("Playwright is not installed. Please install it with: pip install playwright && playwright install")

@st.cache_resource
def load_sentiment_pipeline():
    """Load the sentiment analysis pipeline"""
    try:
        return pipeline("text-classification", model="ProsusAI/finbert")
    except Exception as e:
        st.error(f"Error loading sentiment pipeline: {e}")
        return None

def scrape_yahoo_tickers():
    """Scrape Yahoo Finance for trending tickers"""
    url = 'https://finance.yahoo.com/topic/stock-market-news/'
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
    except requests.RequestException as e:
        st.error(f"Error fetching Yahoo Finance page: {e}")
        return None, None, None, None, None, None
    
    tickers = []
    
    # Scrape tickers
    for a in soup.find_all('a', href=True):
        href = a['href']
        if href.startswith('/quote/') and href.count('/') >= 2:
            ticker = href.split('/')[2]
            if ticker.isupper() and 1 <= len(ticker) <= 5:
                tickers.append(ticker.strip('/'))
    
    ticker_counts = Counter(tickers)
    
    # Initialize variables
    top1_ticker = top2_ticker = top3_ticker = None
    top1_keyword = top2_keyword = top3_keyword = None
    
    if ticker_counts:
        most_common = ticker_counts.most_common(10)
        
        # Display ticker counts in expander
        with st.expander("View Ticker Mention Counts"):
            st.write("Top mentioned tickers by count:")
            for ticker, count in most_common:
                st.write(f"{ticker}: {count} mentions")
        
        # Get all tickers with their counts
        all_tickers_with_counts = ticker_counts.most_common()
        
        # Build final ranking considering ties
        final_ranking = []
        used_tickers = set()
        
        # First, add the clear #1 (highest mention count)
        if all_tickers_with_counts:
            top_count = all_tickers_with_counts[0][1]
            top_tickers = [ticker for ticker, count in all_tickers_with_counts if count == top_count]
            
            if len(top_tickers) == 1:
                # Clear winner
                final_ranking.append(top_tickers[0])
                used_tickers.add(top_tickers[0])
            else:
                # Tie for #1 - use market cap to break tie
                market_caps = {}
                for ticker in top_tickers:
                    try:
                        info = yf.Ticker(ticker).info
                        market_cap = info.get('marketCap', 0)
                        market_caps[ticker] = market_cap if market_cap else 0
                    except Exception as e:
                        market_caps[ticker] = 0
                
                sorted_top = sorted(top_tickers, key=lambda x: market_caps[x], reverse=True)
                final_ranking.extend(sorted_top)
                used_tickers.update(sorted_top)
        
        # Now fill remaining spots up to 3 total
        remaining_spots = 3 - len(final_ranking)
        if remaining_spots > 0:
            # Get remaining tickers (those not already used)
            remaining_tickers = [(ticker, count) for ticker, count in all_tickers_with_counts 
                               if ticker not in used_tickers]
            
            if remaining_tickers:
                # Group remaining tickers by count
                remaining_by_count = {}
                for ticker, count in remaining_tickers:
                    if count not in remaining_by_count:
                        remaining_by_count[count] = []
                    remaining_by_count[count].append(ticker)
                
                # Process each count level
                for count in sorted(remaining_by_count.keys(), reverse=True):
                    tickers_with_count = remaining_by_count[count]
                    
                    if len(tickers_with_count) <= remaining_spots:
                        # Can fit all tickers with this count
                        if len(tickers_with_count) > 1:
                            # Tie - use market cap
                            market_caps = {}
                            for ticker in tickers_with_count:
                                try:
                                    info = yf.Ticker(ticker).info
                                    market_cap = info.get('marketCap', 0)
                                    market_caps[ticker] = market_cap if market_cap else 0
                                except Exception as e:
                                    market_caps[ticker] = 0
                            
                            sorted_tied = sorted(tickers_with_count, 
                                               key=lambda x: market_caps[x], 
                                               reverse=True)
                            final_ranking.extend(sorted_tied)
                        else:
                            final_ranking.extend(tickers_with_count)
                        
                        remaining_spots -= len(tickers_with_count)
                    else:
                        # More tickers than spots - need to pick best ones by market cap
                        market_caps = {}
                        for ticker in tickers_with_count:
                            try:
                                info = yf.Ticker(ticker).info
                                market_cap = info.get('marketCap', 0)
                                market_caps[ticker] = market_cap if market_cap else 0
                            except Exception as e:
                                market_caps[ticker] = 0
                        
                        sorted_tied = sorted(tickers_with_count, 
                                           key=lambda x: market_caps[x], 
                                           reverse=True)
                        final_ranking.extend(sorted_tied[:remaining_spots])
                        remaining_spots = 0
                    
                    if remaining_spots <= 0:
                        break
        
        # Display market cap info for final top 3 selections
        with st.expander("View Market Cap Rankings (Tie-breakers Only)"):
            st.write("Final top 3 selection process:")
            st.write("Market cap used for tie-breaking among equal mention counts:")
            
            # Show market caps for tickers that were considered in tie-breaking
            tie_breaker_tickers = final_ranking[:3]
            market_caps = {}
            for ticker in tie_breaker_tickers:
                try:
                    info = yf.Ticker(ticker).info
                    market_cap = info.get('marketCap', 0)
                    market_caps[ticker] = market_cap if market_cap else 0
                except Exception as e:
                    market_caps[ticker] = 0
            
            for i, ticker in enumerate(tie_breaker_tickers, 1):
                cap = market_caps[ticker]
                mentions = ticker_counts[ticker]
                st.write(f"#{i}: {ticker} - {mentions} mentions, Market Cap = ${cap:,}")
        
        # Assign final top 3
        top1_ticker = final_ranking[0] if len(final_ranking) > 0 else None
        top2_ticker = final_ranking[1] if len(final_ranking) > 1 else None
        top3_ticker = final_ranking[2] if len(final_ranking) > 2 else None
        
        # Get company names with error handling
        try:
            top1_keyword = yf.Ticker(top1_ticker).info.get('longName', top1_ticker) if top1_ticker else None
        except:
            top1_keyword = top1_ticker
        
        try:
            top2_keyword = yf.Ticker(top2_ticker).info.get('longName', top2_ticker) if top2_ticker else None
        except:
            top2_keyword = top2_ticker
        
        try:
            top3_keyword = yf.Ticker(top3_ticker).info.get('longName', top3_ticker) if top3_ticker else None
        except:
            top3_keyword = top3_ticker
    
    return top1_ticker, top2_ticker, top3_ticker, top1_keyword, top2_keyword, top3_keyword

def analyze_ticker_sentiment(ticker, keyword, pipe):
    """Analyze sentiment for a ticker using Yahoo Finance RSS feed"""
    if not ticker or not keyword or not pipe:
        return {
            'ticker': ticker,
            'company_name': keyword,
            'overall_sentiment': 'Neutral',
            'final_score': 0,
            'num_articles': 0,
            'articles': []
        }
    
    total_score = 0
    num_articles = 0
    articles_data = []
    
    try:
        rss_url = f'https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US'
        feed = feedparser.parse(rss_url)
    except Exception as e:
        return {
            'ticker': ticker,
            'company_name': keyword,
            'overall_sentiment': 'Neutral',
            'final_score': 0,
            'num_articles': 0,
            'articles': []
        }

    search_terms = [
        keyword.lower(),
        keyword.lower().split()[0] if ' ' in keyword else keyword.lower(),
        ticker.lower()
    ]

    for entry in feed.entries:
        try:
            title = entry.title.lower()
            summary = getattr(entry, 'summary', '').lower()

            if not any(term in title or term in summary for term in search_terms):
                continue

            # Use summary for sentiment analysis, fallback to title if no summary
            text_for_analysis = getattr(entry, 'summary', '') or entry.title
            
            sentiment = pipe(text_for_analysis)[0]
            label = sentiment["label"].lower()
            score = sentiment["score"]

            article_info = {
                'title': entry.title,
                'link': getattr(entry, "link", "N/A"),
                'published': getattr(entry, "published", "N/A"),
                'summary': getattr(entry, "summary", "N/A"),
                'sentiment': label.capitalize(),
                'score': score
            }
            articles_data.append(article_info)

            if label == "positive":
                total_score += score
            elif label == "negative":
                total_score -= score

            num_articles += 1
        except Exception as e:
            continue

    if num_articles == 0:
        final_score = 0
        overall = "Neutral"
    else:
        final_score = total_score / num_articles
        if final_score >= 0.15:
            overall = "Positive"
        elif final_score <= -0.15:
            overall = "Negative"
        else:
            overall = "Neutral"

    return {
        'ticker': ticker,
        'company_name': keyword,
        'overall_sentiment': overall,
        'final_score': final_score,
        'num_articles': num_articles,
        'articles': articles_data
    }

def setup_event_loop():
    """Setup proper event loop for different platforms"""
    if platform.system() == 'Windows':
        # Use ProactorEventLoop on Windows for better subprocess support
        if hasattr(asyncio, 'WindowsProactorEventLoopPolicy'):
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    # Create new event loop if none exists or if current loop is closed
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("Event loop is closed")
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop

async def scrape_and_classify_user_tweets(username, tickers_keywords, max_tweets=30, delay=2):
    """Scrape tweets from a user and classify them by ticker mentions"""
    results = {ticker: [] for ticker, _ in tickers_keywords if ticker}

    browser = None
    try:
        async with async_playwright() as p:
            # Try browsers in order of compatibility
            browser_types = [p.chromium, p.firefox]
            
            for browser_type in browser_types:
                try:
                    browser = await browser_type.launch(
                        headless=True,
                        args=[
                            '--no-sandbox',
                            '--disable-dev-shm-usage',
                            '--disable-gpu',
                            '--disable-extensions',
                            '--disable-background-timer-throttling',
                            '--disable-backgrounding-occluded-windows',
                            '--disable-renderer-backgrounding'
                        ] if browser_type == p.chromium else []
                    )
                    break
                except Exception as e:
                    continue
            
            if not browser:
                raise Exception("Could not launch any browser")
            
            context = await browser.new_context(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            )
            page = await context.new_page()
            
            # Set reasonable timeout
            page.set_default_timeout(30000)
            
            # Navigate to Twitter profile
            url = f"https://twitter.com/{username}"
            await page.goto(url, wait_until='networkidle')
            await page.wait_for_timeout(3000)

            processed_tweets = set()
            scroll_attempts = 0
            max_scrolls = 10

            while scroll_attempts < max_scrolls:
                try:
                    # Look for tweet articles
                    tweet_articles = await page.query_selector_all('article')

                    for article in tweet_articles:
                        try:
                            # Try to find tweet text element
                            tweet_element = await article.query_selector('div[data-testid="tweetText"]')
                            if tweet_element:
                                tweet_text = await tweet_element.inner_text()

                                if tweet_text in processed_tweets or not tweet_text:
                                    continue 

                                processed_tweets.add(tweet_text)
                                text_lower = tweet_text.lower()

                                # Check if tweet mentions any of our tickers
                                for ticker, keyword in tickers_keywords:
                                    if ticker and keyword:  # Ensure both are not None
                                        ticker_variants = [
                                            f"${ticker.lower()}",
                                            ticker.lower(),
                                            keyword.lower()
                                        ]
                                        
                                        if any(variant in text_lower for variant in ticker_variants):
                                            if len(results[ticker]) < max_tweets:
                                                results[ticker].append(tweet_text)
                        except Exception as e:
                            continue

                    # Scroll down
                    await page.mouse.wheel(0, 10000)
                    scroll_attempts += 1
                    await page.wait_for_timeout(delay * 1000)
                    
                except Exception as e:
                    break

    except Exception as e:
        st.error(f"Error scraping @{username}: {str(e)}")
        
    finally:
        if browser:
            try:
                await browser.close()
            except:
                pass

    return results


async def analyze_ticker_tweets(tweets, pipe):
    """Analyze sentiment of tweets for a ticker"""
    if not tweets or not pipe:
        return 0, "Neutral", 0, []
    
    total_score = 0
    valid_analyses = 0
    detailed_results = []

    for tweet in tweets:
        try:
            # Clean tweet text but keep it readable
            clean_text = re.sub(r'http\S+|www\S+|https\S+', '[URL]', tweet, flags=re.MULTILINE)
            
            if len(clean_text) < 10:  # Skip very short tweets
                continue
            
            # Analyze sentiment
            sentiment_result = pipe(clean_text)
            
            # Handle if pipe returns a list or single result
            if isinstance(sentiment_result, list):
                sentiment_result = sentiment_result[0]
            
            label = sentiment_result["label"].lower()
            confidence = sentiment_result["score"]
            
            # Convert to our scoring system
            if "neg" in label or label == "negative":
                score = -confidence
            elif "pos" in label or label == "positive":
                score = confidence
            else:  # neutral
                score = 0
            
            total_score += score
            valid_analyses += 1
            
            # Store detailed result
            detailed_results.append({
                'text': tweet,
                'clean_text': clean_text,
                'sentiment': label.capitalize(),
                'score': score,
                'confidence': confidence
            })
            
        except Exception as e:
            # Store failed analysis for debugging
            detailed_results.append({
                'text': tweet,
                'clean_text': 'Analysis failed',
                'sentiment': 'Error',
                'score': 0,
                'confidence': 0,
                'error': str(e)
            })
            continue

    if valid_analyses == 0:
        return 0, "Neutral", 0, detailed_results

    avg_score = total_score / valid_analyses
    
    # Determine overall sentiment
    if avg_score >= 0.15:
        overall_sentiment = "Positive"
    elif avg_score <= -0.15:
        overall_sentiment = "Negative"
    else:
        overall_sentiment = "Neutral"

    return avg_score, overall_sentiment, valid_analyses, detailed_results


async def aggregate_twitter_sentiment(usernames, tickers_keywords, pipe, max_tweets=30):
    """Aggregate Twitter sentiment across multiple users"""
    all_ticker_tweets = {ticker: [] for ticker, _ in tickers_keywords if ticker}

    for i, username in enumerate(usernames):
        st.write(f"Processing @{username}... ({i+1}/{len(usernames)})")
        
        try:
            user_tweets = await scrape_and_classify_user_tweets(username, tickers_keywords, max_tweets)
            
            # Aggregate tweets by ticker
            for ticker in all_ticker_tweets:
                if ticker in user_tweets and user_tweets[ticker]:
                    all_ticker_tweets[ticker].extend(user_tweets[ticker])
                    
        except Exception as e:
            st.warning(f"Failed to process @{username}: {str(e)}")
            continue

    # Analyze sentiment for each ticker
    results = {}
    detailed_twitter_data = {}
    
    for ticker, tweets in all_ticker_tweets.items():
        if tweets and len(tweets) > 0:
            score, sentiment, num_analyzed, detailed_results = await analyze_ticker_tweets(tweets, pipe)
            results[ticker] = {
                'score': score,
                'sentiment': sentiment,
                'num_tweets': len(tweets),
                'num_analyzed': num_analyzed
            }
            detailed_twitter_data[ticker] = detailed_results
        else:
            results[ticker] = {
                'score': 0,
                'sentiment': 'Neutral',
                'num_tweets': 0,
                'num_analyzed': 0
            }
            detailed_twitter_data[ticker] = []
    
    # Store detailed data in session state for the expanders
    st.session_state.detailed_twitter_data = detailed_twitter_data
    
    return results


def run_twitter_analysis(usernames, tickers_keywords, pipe):
    """Run Twitter analysis with proper event loop handling"""
    if not PLAYWRIGHT_AVAILABLE:
        st.error("Playwright is not available. Please install it first.")
        return {}
    
    try:
        # Setup event loop
        loop = setup_event_loop()
        
        # Run the analysis
        results = loop.run_until_complete(
            aggregate_twitter_sentiment(usernames, tickers_keywords, pipe)
        )
        
        return results
        
    except Exception as e:
        st.error(f"Twitter analysis failed: {str(e)}")
        st.error("This might be due to Twitter's anti-bot measures or network issues.")
        return {}

def main():
    st.set_page_config(page_title="Stock Sentiment Analyzer", page_icon="📈", layout="wide")
    
    st.title("Stock Sentiment Analyzer")
    st.markdown("Analyze sentiment for trending stocks from Yahoo Finance news and Twitter")
    
    # Load sentiment pipeline
    pipe = load_sentiment_pipeline()
    if not pipe:
        st.error("Failed to load sentiment analysis pipeline. Please check your internet connection and try again.")
        return
    
    # Create tabs
    tab1, tab2 = st.tabs(["Yahoo Finance Sentiment", "Twitter Sentiment"])
    
    with tab1:
        st.header("Yahoo Finance News Sentiment Analysis")
        
        if st.button("Analyze Yahoo Finance Sentiment", key="yahoo_btn"):
            with st.spinner("Scraping trending tickers from Yahoo Finance..."):
                top1_ticker, top2_ticker, top3_ticker, top1_keyword, top2_keyword, top3_keyword = scrape_yahoo_tickers()
            
            if all([top1_ticker, top2_ticker, top3_ticker]):
                st.success("Successfully identified top 3 trending tickers!")
                
                # Display top tickers
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("#1 Ticker", top1_ticker, top1_keyword)
                with col2:
                    st.metric("#2 Ticker", top2_ticker, top2_keyword)
                with col3:
                    st.metric("#3 Ticker", top3_ticker, top3_keyword)
                
                tickers_keywords = [
                    (top1_ticker, top1_keyword),
                    (top2_ticker, top2_keyword),
                    (top3_ticker, top3_keyword)
                ]
                
                st.subheader("Sentiment Analysis Results")
                
                results = []
                progress_bar = st.progress(0)
                
                for i, (ticker, keyword) in enumerate(tickers_keywords):
                    with st.spinner(f"Analyzing sentiment for {ticker}..."):
                        result = analyze_ticker_sentiment(ticker, keyword, pipe)
                        results.append(result)
                    progress_bar.progress((i + 1) / len(tickers_keywords))
                
                # Create results DataFrame
                df = pd.DataFrame(results)
                
                # Display results in a nice table
                st.dataframe(
                    df[['ticker', 'company_name', 'overall_sentiment', 'final_score', 'num_articles']],
                    column_config={
                        'ticker': 'Ticker',
                        'company_name': 'Company Name',
                        'overall_sentiment': 'Sentiment',
                        'final_score': st.column_config.NumberColumn('Score', format="%.3f"),
                        'num_articles': 'Articles Found'
                    },
                    use_container_width=True
                )
                
                # Visualize sentiment scores using Plotly
                st.subheader("Sentiment Score Visualization")
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=[r['ticker'] for r in results],
                        y=[r['final_score'] for r in results],
                        text=[f"{r['final_score']:.3f}" for r in results],
                        textposition='auto',
                        marker_color=['green' if r['final_score'] > 0.15 else 'red' if r['final_score'] < -0.15 else 'gray' for r in results]
                    )
                ])
                
                fig.update_layout(
                    title="Yahoo Finance Sentiment Scores",
                    xaxis_title="Ticker",
                    yaxis_title="Sentiment Score",
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show detailed articles in expanders
                st.subheader("Detailed Article Analysis")
                for result in results:
                    with st.expander(f"Articles for {result['ticker']} - {result['overall_sentiment']} ({result['final_score']:.3f})"):
                        for article in result['articles']:
                            st.write(f"**Title:** {article['title']}")
                            st.write(f"**Link:** {article['link']}")
                            st.write(f"**Published:** {article['published']}")
                            st.write(f"**Summary:** {article['summary']}")
                            st.write(f"**Sentiment:** {article['sentiment']} (Score: {article['score']:.2f})")
                            st.write("---")
                
                # Store results in session state for Twitter analysis
                st.session_state.yahoo_results = results
                st.session_state.tickers_keywords = tickers_keywords
            else:
                st.error("Failed to identify trending tickers. Please try again.")
    
    with tab2:
        st.header("Twitter Sentiment Analysis")
        
        # Default Twitter usernames
        default_usernames = ["TrendSpider", "AnthonySandford", "StockSavvyShay", "Beth_Kindig", 
                           "ronjonbSaaS", "Jake_Wujastyk", "zerohedge", "Jayendra_22"]
        
        # Allow user to customize Twitter usernames
        st.subheader("Twitter Accounts to Analyze")
        usernames_input = st.text_area(
            "Enter Twitter usernames (one per line, without @):",
            value="\n".join(default_usernames),
            height=150
        )
        
        usernames = [username.strip() for username in usernames_input.split('\n') if username.strip()]
        
        st.write(f"Will analyze {len(usernames)} Twitter accounts")
        
        if not PLAYWRIGHT_AVAILABLE:
            st.error("Playwright is required for Twitter analysis. Please install it with:")
            st.code("pip install playwright && playwright install")
        
        if st.button("Analyze Twitter Sentiment", key="twitter_btn"):
            if 'tickers_keywords' not in st.session_state:
                st.warning("Please run Yahoo Finance analysis first to identify trending tickers!")
            else:
                tickers_keywords = st.session_state.tickers_keywords
                
                with st.spinner("Analyzing Twitter sentiment... This may take several minutes..."):
                    twitter_results = run_twitter_analysis(usernames, tickers_keywords, pipe)
                
                if twitter_results and any(result['num_tweets'] > 0 for result in twitter_results.values()):
                    st.success("Twitter sentiment analysis completed!")
                    
                    # Display Twitter results
                    st.subheader("Twitter Sentiment Results")
                    
                    twitter_df_data = []
                    for ticker, result in twitter_results.items():
                        # Find company name from tickers_keywords
                        company_name = next((keyword for t, keyword in tickers_keywords if t == ticker), ticker)
                        twitter_df_data.append({
                            'ticker': ticker,
                            'company_name': company_name,
                            'sentiment': result['sentiment'],
                            'score': result['score'],
                            'num_tweets': result['num_tweets'],
                            'analyzed': result['num_analyzed']
                        })
                    
                    twitter_df = pd.DataFrame(twitter_df_data)
                    
                    st.dataframe(
                        twitter_df,
                        column_config={
                            'ticker': 'Ticker',
                            'company_name': 'Company Name',
                            'sentiment': 'Sentiment',
                            'score': st.column_config.NumberColumn('Score', format="%.3f"),
                            'num_tweets': 'Tweets Found',
                            'analyzed': 'Tweets Analyzed'
                        },
                        use_container_width=True
                    )
                    
                    # Show detailed tweets in expanders (similar to Yahoo Finance articles)
                    st.subheader("Detailed Tweet Analysis")
                    
                    # Store detailed tweet data for display
                    if 'detailed_twitter_data' not in st.session_state:
                        st.session_state.detailed_twitter_data = {}
                    
                    for ticker, result in twitter_results.items():
                        company_name = next((keyword for t, keyword in tickers_keywords if t == ticker), ticker)
                        num_tweets = result['num_tweets']
                        
                        if num_tweets > 0:
                            with st.expander(f"Tweets for {ticker} - {company_name} - {result['sentiment']} ({result['score']:.3f}) - {num_tweets} tweets"):
                                if ticker in st.session_state.detailed_twitter_data:
                                    tweets_with_sentiment = st.session_state.detailed_twitter_data[ticker]
                                    for i, tweet_data in enumerate(tweets_with_sentiment):
                                        st.write(f"**Tweet {i+1}:**")
                                        st.write(f"*Text:* {tweet_data['text']}")
                                        st.write(f"*Sentiment:* {tweet_data['sentiment']} (Score: {tweet_data['score']:.3f})")
                                        st.write("---")
                                else:
                                    st.write("Tweet details not available. This may occur if tweets were found but sentiment analysis failed.")
                        else:
                            with st.expander(f"No tweets found for {ticker} - {company_name}"):
                                st.write("No tweets mentioning this ticker were found in the analyzed accounts.")
                    
                    # Visualize Twitter sentiment scores
                    st.subheader("Twitter Sentiment Score Visualization")
                    
                    fig_twitter = go.Figure(data=[
                        go.Bar(
                            x=[data['ticker'] for data in twitter_df_data],
                            y=[data['score'] for data in twitter_df_data],
                            text=[f"{data['score']:.3f}" for data in twitter_df_data],
                            textposition='auto',
                            marker_color=['green' if data['score'] > 0.05 else 'red' if data['score'] < -0.05 else 'gray' for data in twitter_df_data],
                            name='Twitter Sentiment'
                        )
                    ])
                    
                    fig_twitter.update_layout(
                        title="Twitter Sentiment Scores",
                        xaxis_title="Ticker",
                        yaxis_title="Sentiment Score",
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig_twitter, use_container_width=True)
                    
                    # Compare Yahoo vs Twitter sentiment if both exist
                    if 'yahoo_results' in st.session_state:
                        st.subheader("Yahoo Finance vs Twitter Sentiment Comparison")
                        
                        # Create comparison chart
                        comparison_data = []
                        for yahoo_result in st.session_state.yahoo_results:
                            ticker = yahoo_result['ticker']
                            yahoo_score = yahoo_result['final_score']
                            twitter_score = twitter_results.get(ticker, {}).get('score', 0)
                            
                            comparison_data.append({
                                'Ticker': ticker,
                                'Yahoo Finance': yahoo_score,
                                'Twitter': twitter_score
                            })
                        
                        comparison_df = pd.DataFrame(comparison_data)
                        
                        fig_comparison = go.Figure()
                        
                        fig_comparison.add_trace(go.Bar(
                            name='Yahoo Finance',
                            x=comparison_df['Ticker'],
                            y=comparison_df['Yahoo Finance'],
                            marker_color='lightblue'
                        ))
                        
                        fig_comparison.add_trace(go.Bar(
                            name='Twitter',
                            x=comparison_df['Ticker'],
                            y=comparison_df['Twitter'],
                            marker_color='lightcoral'
                        ))
                        
                        fig_comparison.update_layout(
                            title='Sentiment Comparison: Yahoo Finance vs Twitter',
                            xaxis_title='Ticker',
                            yaxis_title='Sentiment Score',
                            barmode='group'
                        )
                        
                        st.plotly_chart(fig_comparison, use_container_width=True)
                        
                        # Summary table
                        st.subheader("Summary Comparison")
                        st.dataframe(comparison_df, use_container_width=True)
                else:
                    st.error("No Twitter data was successfully retrieved. This could be due to:")
                    st.write("- Twitter's anti-bot protection")
                    st.write("- Network connectivity issues") 
                    st.write("- The accounts being private or suspended")
                    st.write("- Rate limiting from Twitter")

if __name__ == "__main__":
    main()