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


class FormationAdvisor:
    """Agent that recommends a football formation based on telemetry data."""

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
                temperature=0.4,
            )
            prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    "You are an elite football formation strategist. "
                    "Given a snapshot of real-time player telemetry (speed, fatigue, "
                    "positions), recommend the optimal formation (e.g. 4-3-3, 4-4-2, "
                    "3-5-2) for the next 15 minutes of play. "
                    "Provide:\n"
                    "1. Recommended formation\n"
                    "2. Key reasoning (fatigue distribution, positional imbalances)\n"
                    "3. Specific player movements to execute the transition",
                ),
                ("user", "Telemetry Snapshot:\n{telemetry_data}\n\nCoach Query: {query}"),
            ])
            self._chain = prompt | llm | StrOutputParser()
        return self._chain

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        retry=_should_retry,
        reraise=True,
    )
    async def advise(self, telemetry_data: list, query: str = "What formation should we use?") -> str:
        chain = self._get_chain()
        try:
            return await asyncio.wait_for(
                chain.ainvoke({"telemetry_data": str(telemetry_data), "query": query}),
                timeout=30.0,
            )
        except asyncio.TimeoutError as exc:
            raise HTTPException(status_code=504, detail="Formation advisor timed out.") from exc


formation_advisor = FormationAdvisor()
