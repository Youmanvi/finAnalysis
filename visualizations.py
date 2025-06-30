import streamlit as st
import pandas as pd
import plotly.graph_objects as go

def create_sentiment_chart(results, title="Sentiment Scores"):
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
        title=title,
        xaxis_title="Ticker",
        yaxis_title="Sentiment Score",
        showlegend=False
    )
    
    return fig

def create_twitter_sentiment_chart(twitter_df_data, title="Twitter Sentiment Scores"):
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
        title=title,
        xaxis_title="Ticker",
        yaxis_title="Sentiment Score",
        showlegend=False
    )
    
    return fig_twitter

def create_comparison_chart(yahoo_results, twitter_results):
    comparison_data = []
    for yahoo_result in yahoo_results:
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
    
    return fig_comparison, comparison_df

def display_yahoo_results(results):
    df = pd.DataFrame(results)
    
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

def display_twitter_results(twitter_df_data):
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

def display_article_details(results):
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

def display_tweet_details(twitter_results, tickers_keywords):
    st.subheader("Detailed Tweet Analysis")
    
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