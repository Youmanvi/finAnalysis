import streamlit as st
import requests
import feedparser
import yfinance as yf
from bs4 import BeautifulSoup
from collections import Counter

def scrape_yahoo_tickers():
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
    
    for a in soup.find_all('a', href=True):
        href = a['href']
        if href.startswith('/quote/') and href.count('/') >= 2:
            ticker = href.split('/')[2]
            if ticker.isupper() and 1 <= len(ticker) <= 5:
                tickers.append(ticker.strip('/'))
    
    ticker_counts = Counter(tickers)
    
    top1_ticker = top2_ticker = top3_ticker = None
    top1_keyword = top2_keyword = top3_keyword = None
    
    if ticker_counts:
        most_common = ticker_counts.most_common(10)
        
        with st.expander("View Ticker Mention Counts"):
            st.write("Top mentioned tickers by count:")
            for ticker, count in most_common:
                st.write(f"{ticker}: {count} mentions")
        
        all_tickers_with_counts = ticker_counts.most_common()
        final_ranking = []
        used_tickers = set()
        
        if all_tickers_with_counts:
            top_count = all_tickers_with_counts[0][1]
            top_tickers = [ticker for ticker, count in all_tickers_with_counts if count == top_count]
            
            if len(top_tickers) == 1:
                final_ranking.append(top_tickers[0])
                used_tickers.add(top_tickers[0])
            else:
                market_caps = {}
                for ticker in top_tickers:
                    try:
                        info = yf.Ticker(ticker).info
                        market_cap = info.get('marketCap', 0)
                        market_caps[ticker] = market_cap if market_cap else 0
                    except Exception:
                        market_caps[ticker] = 0
                
                sorted_top = sorted(top_tickers, key=lambda x: market_caps[x], reverse=True)
                final_ranking.extend(sorted_top)
                used_tickers.update(sorted_top)
        
        remaining_spots = 3 - len(final_ranking)
        if remaining_spots > 0:
            remaining_tickers = [(ticker, count) for ticker, count in all_tickers_with_counts 
                               if ticker not in used_tickers]
            
            if remaining_tickers:
                remaining_by_count = {}
                for ticker, count in remaining_tickers:
                    if count not in remaining_by_count:
                        remaining_by_count[count] = []
                    remaining_by_count[count].append(ticker)
                
                for count in sorted(remaining_by_count.keys(), reverse=True):
                    tickers_with_count = remaining_by_count[count]
                    
                    if len(tickers_with_count) <= remaining_spots:
                        if len(tickers_with_count) > 1:
                            market_caps = {}
                            for ticker in tickers_with_count:
                                try:
                                    info = yf.Ticker(ticker).info
                                    market_cap = info.get('marketCap', 0)
                                    market_caps[ticker] = market_cap if market_cap else 0
                                except Exception:
                                    market_caps[ticker] = 0
                            
                            sorted_tied = sorted(tickers_with_count, 
                                               key=lambda x: market_caps[x], 
                                               reverse=True)
                            final_ranking.extend(sorted_tied)
                        else:
                            final_ranking.extend(tickers_with_count)
                        
                        remaining_spots -= len(tickers_with_count)
                    else:
                        market_caps = {}
                        for ticker in tickers_with_count:
                            try:
                                info = yf.Ticker(ticker).info
                                market_cap = info.get('marketCap', 0)
                                market_caps[ticker] = market_cap if market_cap else 0
                            except Exception:
                                market_caps[ticker] = 0
                        
                        sorted_tied = sorted(tickers_with_count, 
                                           key=lambda x: market_caps[x], 
                                           reverse=True)
                        final_ranking.extend(sorted_tied[:remaining_spots])
                        remaining_spots = 0
                    
                    if remaining_spots <= 0:
                        break
        
        with st.expander("View Market Cap Rankings (Tie-breakers Only)"):
            st.write("Final top 3 selection process:")
            st.write("Market cap used for tie-breaking among equal mention counts:")
            
            tie_breaker_tickers = final_ranking[:3]
            market_caps = {}
            for ticker in tie_breaker_tickers:
                try:
                    info = yf.Ticker(ticker).info
                    market_cap = info.get('marketCap', 0)
                    market_caps[ticker] = market_cap if market_cap else 0
                except Exception:
                    market_caps[ticker] = 0
            
            for i, ticker in enumerate(tie_breaker_tickers, 1):
                cap = market_caps[ticker]
                mentions = ticker_counts[ticker]
                st.write(f"#{i}: {ticker} - {mentions} mentions, Market Cap = ${cap:,}")
        
        top1_ticker = final_ranking[0] if len(final_ranking) > 0 else None
        top2_ticker = final_ranking[1] if len(final_ranking) > 1 else None
        top3_ticker = final_ranking[2] if len(final_ranking) > 2 else None
        
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
    except Exception:
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
        except Exception:
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