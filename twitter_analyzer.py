import streamlit as st
import asyncio
import re
from utils import setup_event_loop, check_playwright

PLAYWRIGHT_AVAILABLE = check_playwright()

async def get_browser(p):
    browsers_to_try = ['chromium', 'firefox']  # Only Chromium and Firefox

    for browser_name in browsers_to_try:
        try:
            browser_type = getattr(p, browser_name)
            browser = await browser_type.launch(headless=True)
            print(f"Successfully launched {browser_name}.")
            return browser, browser_name
        except Exception as e:
            print(f"Failed to launch {browser_name}: {e}")
            continue

    raise Exception("Could not launch Chromium or Firefox.")

async def scrape_and_classify_user_tweets(username, tickers_keywords, max_tweets=30, delay=2):
    results = {ticker: [] for ticker, _ in tickers_keywords if ticker}

    try:
        from playwright.async_api import async_playwright

        async with async_playwright() as p:
            browser, browser_name = await get_browser(p)

            context = await browser.new_context()  # Default context
            page = await context.new_page()
            page.set_default_timeout(30000)

            print(f"Using {browser_name} to scrape @{username}")

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
                                        ticker_variants = [
                                            f"${ticker.lower()}",
                                            ticker.lower(),
                                            keyword.lower()
                                        ]

                                        if any(variant in text_lower for variant in ticker_variants):
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
