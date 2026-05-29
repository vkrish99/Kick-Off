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

# Retry on any exception except HTTPException (configuration errors should surface immediately)
_should_retry = retry_if_exception(lambda e: not isinstance(e, HTTPException))


class TacticalAgent:
    """Analyses player telemetry and answers tactical questions."""

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
                model=settings.GROQ_MODEL,
                api_key=settings.GROQ_API_KEY,
                temperature=0.3,
            )
            prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    "You are a world-class football tactical analyst. "
                    "Analyse the provided telemetry data and answer the user's tactical "
                    "question. Focus on spacing, speed, positioning, and formation dynamics.",
                ),
                ("user", "Telemetry Data:\n{telemetry_data}\n\nQuestion: {query}"),
            ])
            self._chain = prompt | llm | StrOutputParser()
        return self._chain

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        retry=_should_retry,
        reraise=True,
    )
    async def analyze(self, telemetry_data: list, query: str) -> str:
        chain = self._get_chain()
        try:
            return await asyncio.wait_for(
                chain.ainvoke({"telemetry_data": str(telemetry_data), "query": query}),
                timeout=30.0,
            )
        except asyncio.TimeoutError as exc:
            raise HTTPException(status_code=504, detail="Tactical agent timed out.") from exc


tactical_agent = TacticalAgent()
