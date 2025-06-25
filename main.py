import streamlit as st
import pandas as pd
from utils import load_sentiment_pipeline, check_playwright
from yahoo_analyzer import scrape_yahoo_tickers, analyze_ticker_sentiment
from twitter_analyzer import run_twitter_analysis
from visualizations import (
    create_sentiment_chart, create_twitter_sentiment_chart, 
    create_comparison_chart, display_yahoo_results, 
    display_twitter_results, display_article_details, 
    display_tweet_details
)

def main():
    st.set_page_config(page_title="Stock Sentiment Analyzer", page_icon="📈", layout="wide")
    
    st.title("Stock Sentiment Analyzer")
    st.markdown("Analyze sentiment for trending stocks from Yahoo Finance news and Twitter")
    
    pipe = load_sentiment_pipeline()
    if not pipe:
        st.error("Can't load pipeline")
        return
    
    tab1, tab2 = st.tabs(["Yahoo Finance Sentiment", "Twitter Sentiment"])
    
    with tab1:
        st.header("Yahoo Finance News Sentiment Analysis")
        
        if st.button("Analyze Yahoo Finance Sentiment", key="yahoo_btn"):
            with st.spinner("Scraping trending tickers from Yahoo Finance..."):
                top1_ticker, top2_ticker, top3_ticker, top1_keyword, top2_keyword, top3_keyword = scrape_yahoo_tickers()
                
            if all([top1_ticker, top2_ticker, top3_ticker]):
                st.success("Successfully identified top 3 trending tickers!")
                
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
                
                progress_bar.empty()
                
                # Display results
                display_yahoo_results(results)
                
                # Create and display chart
                fig = create_sentiment_chart(results, "Yahoo Finance Sentiment Analysis")
                st.plotly_chart(fig, use_container_width=True)
                
                # Store results in session state for comparison
                st.session_state.yahoo_results = results
                st.session_state.tickers_keywords = tickers_keywords
                
                # Display detailed article analysis
                display_article_details(results)
            else:
                st.error("Analysis failed")
    
    with tab2:
        st.header("Twitter Sentiment Analysis")
        
        if not check_playwright():
            st.warning("Playwright Error")
        else:
            st.info("Enter Twitter Usernames.")
            
            # Input for Twitter usernames
            default_users = ["TrendSpider", "AnthonySandford", "StockSavvyShay", "Beth_Kindig", 
                           "ronjonbSaaS", "Jake_Wujastyk", "zerohedge", "Jayendra_22"]
            usernames_input = st.text_area(
                "Twitter Usernames (new line):",
                value="\n".join(default_users),
                help="Enter Twitter usernames"
            )
            
            usernames = [username.strip() for username in usernames_input.split('\n') if username.strip()]
            
            if st.button("Analyze Twitter Sentiment", key="twitter_btn"):
                if not usernames:
                    st.error("Please enter at least one Twitter username.")
                elif 'tickers_keywords' not in st.session_state:
                    st.error("Please run Yahoo Finance analysis first to identify trending tickers.")
                else:
                    tickers_keywords = st.session_state.tickers_keywords
                    
                    with st.spinner("Analyzing Twitter sentiments."):
                        twitter_results = run_twitter_analysis(usernames, tickers_keywords, pipe)
                    
                    if twitter_results:
                        st.success("Twitter sentiment analysis completed!")
                        
                        # Prepare data for display
                        twitter_df_data = []
                        for ticker, result in twitter_results.items():
                            company_name = next((keyword for t, keyword in tickers_keywords if t == ticker), ticker)
                            twitter_df_data.append({
                                'ticker': ticker,
                                'company_name': company_name,
                                'sentiment': result['sentiment'],
                                'score': result['score'],
                                'num_tweets': result['num_tweets'],
                                'analyzed': result['num_analyzed']
                            })
                        
                        # Display Twitter results
                        display_twitter_results(twitter_df_data)
                        
                        # Create and display Twitter chart
                        fig_twitter = create_twitter_sentiment_chart(twitter_df_data, "Twitter Sentiment Analysis")
                        st.plotly_chart(fig_twitter, use_container_width=True)
                        
                        # Store Twitter results for comparison
                        st.session_state.twitter_results = twitter_results
                        
                        # Display detailed tweet analysis
                        display_tweet_details(twitter_results, tickers_keywords)
                        
                        # Comparison chart if Yahoo results are available
                        if 'yahoo_results' in st.session_state:
                            st.subheader("Sentiment Comparison")
                            fig_comparison, comparison_df = create_comparison_chart(
                                st.session_state.yahoo_results, 
                                twitter_results
                            )
                            st.plotly_chart(fig_comparison, use_container_width=True)
                            
                            st.subheader("Comparison Data")
                            st.dataframe(comparison_df, use_container_width=True)
                    else:
                        st.error("Twitter analysis failed.")

if __name__ == "__main__":
    main()