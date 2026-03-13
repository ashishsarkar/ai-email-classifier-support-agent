"""Test script to demonstrate FAISS KB integration in the workflow."""

import asyncio
import sys

# Add parent directory to path
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

from src.services.vector_kb_service import VectorKBService
from src.services.llm_service import LLMService
from loguru import logger


async def test_kb_integration():
    """Test FAISS knowledge base integration."""
    try:
        logger.info("=" * 80)
        logger.info("FAISS Knowledge Base Integration Test")
        logger.info("=" * 80 + "\n")

        # Initialize services
        logger.info("1️⃣  Initializing FAISS Vector KB...")
        vector_kb = VectorKBService()
        await vector_kb.initialize()

        health = await vector_kb.health_check()
        logger.info(f"   Status: {health['status']}")
        logger.info(f"   Documents in KB: {health['documents_count']}\n")

        # Test scenario 1: Billing question
        logger.info("2️⃣  Test Scenario 1: Customer asking about billing\n")
        customer_email_1 = {
            "subject": "How much does CloudSync cost?",
            "body": "Hi, I want to know about your pricing plans and if there are any discounts available for annual subscriptions.",
        }

        logger.info(f"Customer Question: {customer_email_1['subject']}")
        logger.info(f"Details: {customer_email_1['body']}\n")

        # Search KB
        search_query = f"{customer_email_1['subject']} {customer_email_1['body']}"
        kb_results = await vector_kb.search(search_query, category="billing", limit=3, threshold=0.3)

        logger.info(f"🔍 Found {len(kb_results)} relevant documents:\n")
        for i, doc in enumerate(kb_results, 1):
            logger.info(f"   {i}. {doc['title']}")
            logger.info(f"      Relevance: {doc['similarity_score']:.1%}")
            logger.info(f"      Preview: {doc['content'][:150]}...\n")

        # Format context
        context = await vector_kb.format_context(kb_results)
        logger.info("📋 Formatted Context for LLM:")
        logger.info("-" * 80)
        logger.info(context)
        logger.info("-" * 80 + "\n")

        # Test scenario 2: Technical support question
        logger.info("3️⃣  Test Scenario 2: Customer having technical issues\n")
        customer_email_2 = {
            "subject": "Files not syncing between devices",
            "body": "My files are not syncing properly between my laptop and mobile device. What should I do?",
        }

        logger.info(f"Customer Question: {customer_email_2['subject']}")
        logger.info(f"Details: {customer_email_2['body']}\n")

        # Search KB
        search_query = f"{customer_email_2['subject']} {customer_email_2['body']}"
        kb_results = await vector_kb.search(search_query, category="technical_support", limit=3, threshold=0.3)

        logger.info(f"🔍 Found {len(kb_results)} relevant documents:\n")
        for i, doc in enumerate(kb_results, 1):
            logger.info(f"   {i}. {doc['title']}")
            logger.info(f"      Relevance: {doc['similarity_score']:.1%}\n")

        # Test scenario 3: General inquiry
        logger.info("4️⃣  Test Scenario 3: General product inquiry\n")
        customer_email_3 = {
            "subject": "What features does CloudSync have?",
            "body": "I'm interested in learning more about CloudSync Pro features and capabilities.",
        }

        logger.info(f"Customer Question: {customer_email_3['subject']}\n")

        # Search without category filter for broader results
        search_query = f"{customer_email_3['subject']} {customer_email_3['body']}"
        kb_results = await vector_kb.search(search_query, limit=5, threshold=0.3)

        logger.info(f"🔍 Found {len(kb_results)} relevant documents:\n")
        for i, doc in enumerate(kb_results, 1):
            logger.info(f"   {i}. {doc['title']} ({doc['category']})")
            logger.info(f"      Relevance: {doc['similarity_score']:.1%}\n")

        # Test scenario 4: Demonstrate vector similarity search
        logger.info("5️⃣  Vector Similarity Demonstration\n")

        logger.info("Query variations and their top result:\n")
        test_queries = [
            "How do I install CloudSync?",
            "Storage limits and quotas",
            "Enable two-factor authentication",
            "Can I export my data?",
        ]

        for query in test_queries:
            results = await vector_kb.search(query, limit=1)
            if results:
                doc = results[0]
                logger.info(f"   Q: {query}")
                logger.info(f"   → {doc['title']} ({doc['similarity_score']:.1%})\n")

        logger.info("=" * 80)
        logger.info("✅ Knowledge Base Integration Test Complete!")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(test_kb_integration())
