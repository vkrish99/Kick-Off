import asyncio
import logging
from fastapi import HTTPException
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential
from app.agents.tactical import tactical_agent
from app.agents.sentiment import sentiment_agent
from app.agents.formation import formation_advisor
from app.services.rag import rag_service
from app.core.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

_should_retry = retry_if_exception(lambda e: not isinstance(e, HTTPException))


class Orchestrator:
    """Routes natural-language queries to the appropriate specialist agent."""

    def __init__(self):
        self._router_chain = None

    def _get_router_chain(self) -> StrOutputParser:
        if not settings.GROQ_API_KEY:
            raise HTTPException(
                status_code=503,
                detail="GROQ_API_KEY is not configured. Set it in .env.",
            )
        if self._router_chain is None:
            llm = ChatGroq(
                model=settings.GROQ_FAST_MODEL,
                api_key=settings.GROQ_API_KEY,
                temperature=0,
            )
            router_prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    "You are the head coach assistant. Route the user's query to the "
                    "correct specialist by replying with exactly ONE of these tokens:\n"
                    "  TACTICAL   — match tactics, telemetry, player speed/position\n"
                    "  SENTIMENT  — player morale, news, media sentiment\n"
                    "  FORMATION  — formation advice, shape change, line-up\n"
                    "  RAG        — historical data, player profile lookups\n"
                    "  GENERAL    — anything else\n"
                    "Reply with the token only.",
                ),
                ("user", "{query}"),
            ])
            self._router_chain = router_prompt | llm | StrOutputParser()
        return self._router_chain

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        retry=_should_retry,
        reraise=True,
    )
    async def _classify_route(self, query: str) -> str:  # one of: TACTICAL, FORMATION, SENTIMENT, RAG, GENERAL
        """Call the LLM router and return the route token."""
        router_chain = self._get_router_chain()
        try:
            result = await asyncio.wait_for(
                router_chain.ainvoke({"query": query}),
                timeout=15.0,
            )
            return result.strip().upper()
        except asyncio.TimeoutError as exc:
            raise HTTPException(status_code=504, detail="Router timed out.") from exc

    async def process_query(
        self,
        query: str,
        telemetry_context: list = None,
        news_context: str = None,
    ) -> tuple[str, str]:
        """
        Returns (insight, route_used).
        """
        route = await self._classify_route(query)
        logger.info("Orchestrator routed '%s' → %s", query[:60], route)

        if "TACTICAL" in route:
            if telemetry_context:
                result = await tactical_agent.analyze(telemetry_context, query)
                return result, "TACTICAL"
            return (
                "Tactical analysis requires live telemetry data. "
                "Run the pitch simulation first, then re-send your query.",
                "TACTICAL_NO_DATA",
            )

        if "FORMATION" in route:
            if telemetry_context:
                result = await formation_advisor.advise(telemetry_context, query)
                return result, "FORMATION"
            return (
                "Formation advice requires live telemetry data. "
                "Please provide a telemetry_context in your request.",
                "FORMATION_NO_DATA",
            )

        if "SENTIMENT" in route:
            if news_context:
                result = await sentiment_agent.analyze_sentiment(news_context, query)
                return result, "SENTIMENT"
            return (
                "Sentiment analysis requires news context. "
                "Please provide the news_context field in your request.",
                "SENTIMENT_NO_DATA",
            )

        if "RAG" in route:
            docs = await rag_service.retrieve_context(query)
            context_str = "\n".join(d.page_content for d in docs)
            llm = ChatGroq(
                model=settings.GROQ_MODEL,
                api_key=settings.GROQ_API_KEY,
                temperature=0.3,
            )
            synthesis_prompt = ChatPromptTemplate.from_messages([
                ("system", "Answer the query based on the retrieved player profiles."),
                ("user", "Context:\n{context}\n\nQuery: {query}"),
            ])
            chain = synthesis_prompt | llm | StrOutputParser()
            try:
                result = await asyncio.wait_for(
                    chain.ainvoke({"context": context_str, "query": query}),
                    timeout=30.0,
                )
            except asyncio.TimeoutError as exc:
                raise HTTPException(status_code=504, detail="RAG synthesis timed out.") from exc
            return result, "RAG"

        # GENERAL fallback
        return (
            f"Received your query: '{query}'. "
            f"For specialist AI insights, please provide telemetry or news context.",
            "GENERAL",
        )


orchestrator = Orchestrator()
