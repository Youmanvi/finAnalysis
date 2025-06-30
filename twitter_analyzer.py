import streamlit as st
import asyncio
import re
from utils import setup_event_loop, check_playwright

PLAYWRIGHT_AVAILABLE = check_playwright()

def detect_browser():
    """Detect the user's browser from user agent"""
    try:
        # Try to get user agent from streamlit context
        import streamlit.web.server.websocket_headers as wsh
        headers = wsh.get_websocket_headers()
        user_agent = headers.get('user-agent', '').lower()
    except:
        # Fallback - assume Chrome as most common
        user_agent = 'chrome'
    
    if 'firefox' in user_agent or 'gecko' in user_agent:
        return 'firefox'
    elif 'chrome' in user_agent or 'chromium' in user_agent:
        return 'chromium'
    elif 'safari' in user_agent and 'chrome' not in user_agent:
        return 'webkit'
    else:
        return 'chromium'  # Default fallback

async def get_browser(p):
    detected_browser = detect_browser()
    
    browser_configs = {
        'chromium': {
            'headless': True,
            'args': [
                '--no-sandbox',
                '--disable-dev-shm-usage',
                '--disable-gpu',
                '--disable-extensions',
                '--disable-background-timer-throttling',
                '--disable-backgrounding-occluded-windows',
                '--disable-renderer-backgrounding',
                '--disable-web-security',
                '--disable-features=VizDisplayCompositor',
                '--user-data-dir=/tmp/playwright_dev_profile'
            ]
        },
        'firefox': {
            'headless': True,
            'firefox_user_prefs': {
                'dom.webdriver.enabled': False,
                'useAutomationExtension': False,
                'general.platform.override': 'Linux x86_64'
            }
        },
        'webkit': {'headless': True}
    }
    
    # Try detected browser first
    try:
        browser_type = getattr(p, detected_browser)
        browser = await browser_type.launch(**browser_configs[detected_browser])
        return browser, detected_browser
    except Exception as e:
        st.warning(f"Could not launch {detected_browser}, trying fallback browsers...")
    
    # Fallback to other browsers if detected one fails
    fallback_order = ['chromium', 'firefox', 'webkit']
    if detected_browser in fallback_order:
        fallback_order.remove(detected_browser)
    
    for browser_name in fallback_order:
        try:
            browser_type = getattr(p, browser_name)
            browser = await browser_type.launch(**browser_configs[browser_name])
            return browser, browser_name
        except Exception as e:
            continue
    
    raise Exception("Could not launch any browser")

async def scrape_and_classify_user_tweets(username, tickers_keywords, max_tweets=30, delay=2):
    results = {ticker: [] for ticker, _ in tickers_keywords if ticker}

    try:
        from playwright.async_api import async_playwright
        
        async with async_playwright() as p:
            browser, browser_name = await get_browser(p)
            
            context = await browser.new_context(
                user_agent='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                viewport={'width': 1920, 'height': 1080},
                java_script_enabled=True,
                bypass_csp=True,
                ignore_https_errors=True
            )
            
            page = await context.new_page()
            page.set_default_timeout(30000)
            
            url = f"https://twitter.com/{username}"
            await page.goto(url, wait_until='networkidle', timeout=60000)
            await page.wait_for_timeout(5000)

            processed_tweets = set()
            scroll_attempts = 0
            max_scrolls = 10

            while scroll_attempts < max_scrolls:
                try:
                    tweet_articles = await page.query_selector_all('article')

                    for article in tweet_articles:
                        try:
                            tweet_element = await article.query_selector('div[data-testid="tweetText"]')
                            if tweet_element:
                                tweet_text = await tweet_element.inner_text()

                                if tweet_text in processed_tweets or not tweet_text:
                                    continue 

                                processed_tweets.add(tweet_text)
                                text_lower = tweet_text.lower()

                                for ticker, keyword in tickers_keywords:
                                    if ticker and keyword:
                                        # Create more precise matching patterns
                                        ticker_patterns = [
                                            rf'\${ticker.lower()}\b',  # $TICKER with word boundary
                                            rf'\b{re.escape(ticker.lower())}\b',  # TICKER with word boundaries
                                        ]
                                        
                                        # For company names, split into words and check for significant matches
                                        keyword_words = keyword.lower().split()
                                        keyword_patterns = []
                                        
                                        # Add full company name with word boundaries
                                        if len(keyword_words) > 1:
                                            keyword_patterns.append(rf'\b{re.escape(keyword.lower())}\b')
                                        
                                        # Add individual significant words (length > 3 to avoid common words)
                                        for word in keyword_words:
                                            if len(word) > 3 and word not in ['corp', 'inc', 'ltd', 'llc', 'company', 'technologies', 'systems', 'group', 'holdings']:
                                                keyword_patterns.append(rf'\b{re.escape(word)}\b')
                                        
                                        all_patterns = ticker_patterns + keyword_patterns
                                        
                                        # Check if any pattern matches
                                        match_found = False
                                        for pattern in all_patterns:
                                            if re.search(pattern, text_lower):
                                                match_found = True
                                                break
                                        
                                        if match_found:
                                            if len(results[ticker]) < max_tweets:
                                                results[ticker].append(tweet_text)
                        except Exception:
                            continue

                    await page.mouse.wheel(0, 10000)
                    scroll_attempts += 1
                    await page.wait_for_timeout(delay * 1000)
                    
                except Exception:
                    break

            await browser.close()

    except Exception as e:
        st.error(f"Error scraping @{username}: {str(e)}")

    return results

async def analyze_ticker_tweets(tweets, pipe):
    if not tweets or not pipe:
        return 0, "Neutral", 0, []
    
    total_score = 0
    valid_analyses = 0
    detailed_results = []

    for tweet in tweets:
        try:
            clean_text = re.sub(r'http\S+|www\S+|https\S+', '[URL]', tweet, flags=re.MULTILINE)
            
            if len(clean_text) < 10:
                continue
            
            sentiment_result = pipe(clean_text)
            
            if isinstance(sentiment_result, list):
                sentiment_result = sentiment_result[0]
            
            label = sentiment_result["label"].lower()
            confidence = sentiment_result["score"]
            
            if "neg" in label or label == "negative":
                score = -confidence
            elif "pos" in label or label == "positive":
                score = confidence
            else:
                score = 0
            
            total_score += score
            valid_analyses += 1
            
            detailed_results.append({
                'text': tweet,
                'clean_text': clean_text,
                'sentiment': label.capitalize(),
                'score': score,
                'confidence': confidence
            })
            
        except Exception as e:
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
    
    if avg_score >= 0.15:
        overall_sentiment = "Positive"
    elif avg_score <= -0.15:
        overall_sentiment = "Negative"
    else:
        overall_sentiment = "Neutral"

    return avg_score, overall_sentiment, valid_analyses, detailed_results

async def aggregate_twitter_sentiment(usernames, tickers_keywords, pipe, max_tweets=30):
    all_ticker_tweets = {ticker: [] for ticker, _ in tickers_keywords if ticker}

    for i, username in enumerate(usernames):
        st.write(f"Processing @{username}... ({i+1}/{len(usernames)})")
        
        try:
            user_tweets = await scrape_and_classify_user_tweets(username, tickers_keywords, max_tweets)
            
            for ticker in all_ticker_tweets:
                if ticker in user_tweets and user_tweets[ticker]:
                    all_ticker_tweets[ticker].extend(user_tweets[ticker])
                    
        except Exception as e:
            st.warning(f"Failed to process @{username}: {str(e)}")
            continue

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
    
    st.session_state.detailed_twitter_data = detailed_twitter_data
    
    return results

def run_twitter_analysis(usernames, tickers_keywords, pipe):
    if not PLAYWRIGHT_AVAILABLE:
        st.error("Playwright is not available. Please install it first.")
        return {}
    
    try:
        loop = setup_event_loop()
        results = loop.run_until_complete(
            aggregate_twitter_sentiment(usernames, tickers_keywords, pipe)
        )
        return results
        
    except Exception as e:
        st.error(f"Twitter analysis failed: {str(e)}")
        st.error("This might be due to Twitter's anti-bot measures or network issues.")
        return {}