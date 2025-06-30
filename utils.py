import streamlit as st
import asyncio
import platform
from transformers import pipeline

def check_playwright():
    try:
        from playwright.async_api import async_playwright
        return True
    except ImportError:
        return False

@st.cache_resource
def load_sentiment_pipeline():
    try:
        return pipeline("text-classification", model="ProsusAI/finbert")
    except Exception as e:
        st.error(f"Error loading sentiment pipeline: {e}")
        return None

def setup_event_loop():
    if platform.system() == 'Windows':
        if hasattr(asyncio, 'WindowsProactorEventLoopPolicy'):
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("Event loop is closed")
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop