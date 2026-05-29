import asyncio
import logging
from fastapi import HTTPException
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential
from app.core.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

_should_retry = retry_if_exception(lambda e: not isinstance(e, HTTPException))


class SentimentAgent:
    """Analyses news / media content and determines player morale impact."""

    def __init__(self):
        self._chain = None

    def _get_chain(self):
        if not settings.GROQ_API_KEY:
            raise HTTPException(
                status_code=503,
                detail="GROQ_API_KEY is not configured. Set it in .env.",
            )
        if self._chain is None:
            llm = ChatGroq(
                model=settings.GROQ_FAST_MODEL,
                api_key=settings.GROQ_API_KEY,
                temperature=0.3,
            )
            prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    "You are a football psychology and media expert. "
                    "Analyse the news snippets provided and determine the sentiment "
                    "and potential impact on player morale. Be concise and actionable.",
                ),
                ("user", "News Content:\n{news_content}\n\nContext: {context}"),
            ])
            self._chain = prompt | llm | StrOutputParser()
        return self._chain

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        retry=_should_retry,
        reraise=True,
    )
    async def analyze_sentiment(self, news_content: str, context: str = "") -> str:
        chain = self._get_chain()
        try:
            return await asyncio.wait_for(
                chain.ainvoke({"news_content": news_content, "context": context}),
                timeout=30.0,
            )
        except asyncio.TimeoutError as exc:
            raise HTTPException(status_code=504, detail="Sentiment agent timed out.") from exc


sentiment_agent = SentimentAgent()
