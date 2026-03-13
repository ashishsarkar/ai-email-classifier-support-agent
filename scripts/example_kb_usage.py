"""Simple example showing FAISS KB usage in a real scenario."""

import asyncio
import sys

# Add parent directory to path
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

from src.services.vector_kb_service import VectorKBService
from src.services.llm_service import LLMService
from loguru import logger


async def example_workflow():
    """Example: Process customer email with KB-enriched response."""
    try:
        logger.info("=" * 80)
        logger.info("FAISS KB Example: Real Workflow")
        logger.info("=" * 80 + "\n")

        # Initialize services
        vector_kb = VectorKBService()
        await vector_kb.initialize()

        llm_service = LLMService()

        # Step 1: Simulate incoming customer email
        logger.info("📧 Step 1: Customer Email Received\n")

        customer_email = {
            "sender": "john@example.com",
            "subject": "Can't sync files on my phone",
            "body": "Hi, I've been trying to sync my files between my computer and phone, but it's not working. I installed the latest app but nothing is syncing. What should I do?",
        }

        logger.info(f"From: {customer_email['sender']}")
        logger.info(f"Subject: {customer_email['subject']}")
        logger.info(f"Body: {customer_email['body']}\n")

        # Step 2: Extract search query
        logger.info("🔍 Step 2: Extract Search Query\n")

        search_query = f"{customer_email['subject']} {customer_email['body']}"
        logger.info(f"Search Query: {search_query}\n")

        # Step 3: Search knowledge base
        logger.info("📚 Step 3: Search Knowledge Base\n")

        kb_results = await vector_kb.search(
            query=search_query,
            category="technical_support",
            limit=3,
            threshold=0.3
        )

        if kb_results:
            logger.info(f"Found {len(kb_results)} relevant documents:\n")
            for i, doc in enumerate(kb_results, 1):
                logger.info(f"   {i}. {doc['title']}")
                logger.info(f"      Relevance: {doc['similarity_score']:.1%}")
                logger.info(f"      Content: {doc['content'][:100]}...\n")
        else:
            logger.warning("No relevant documents found\n")

        # Step 4: Format context for LLM
        logger.info("📋 Step 4: Format Context for LLM\n")

        context = await vector_kb.format_context(kb_results)
        logger.info("Context to be injected into LLM prompt:")
        logger.info("-" * 80)
        logger.info(context)
        logger.info("-" * 80 + "\n")

        # Step 5: Prepare LLM prompt with context
        logger.info("🤖 Step 5: Prepare LLM Prompt\n")

        prompt_with_context = f"""
You are a helpful customer support agent. Use the following documentation to answer the customer's question.

{context}

Customer Email:
Subject: {customer_email['subject']}
Body: {customer_email['body']}

Please provide a helpful and professional response based on the documentation provided.
"""

        logger.info("Prompt prepared (first 300 chars):")
        logger.info(prompt_with_context[:300] + "...\n")

        # Step 6: Generate response (simulate - don't actually call LLM)
        logger.info("💬 Step 6: Generate Response\n")

        # In real usage, you would call:
        # response = await llm_service.generate_response(
        #     subject=customer_email['subject'],
        #     body=customer_email['body'],
        #     category="technical_support",
        #     priority="medium",
        #     context=context
        # )

        sample_response = """Dear John,

Thank you for contacting us about your sync issue. Based on your description, here are the steps to resolve this:

1. **Check Internet Connection**: Ensure both your computer and phone have a stable internet connection.

2. **Update the App**: Make sure you have the latest version installed. Try clearing the app cache and reinstalling if needed.

3. **Review Permissions**: On your phone, check that CloudSync has permission to access your files in the app settings.

4. **Restart the Sync**: Disable sync, wait 30 seconds, then re-enable it in the app settings.

5. **Check Folder Settings**: Ensure the folders you want to sync are properly selected in the sync settings.

If these steps don't resolve the issue, please try these advanced troubleshooting steps:
- Logout and login again
- Clear the app cache completely
- Reinstall the application

Please let us know if you continue to experience issues!

Best regards,
Customer Support Team"""

        logger.info("Generated Response:")
        logger.info("-" * 80)
        logger.info(sample_response)
        logger.info("-" * 80 + "\n")

        # Step 7: Summary
        logger.info("=" * 80)
        logger.info("Summary")
        logger.info("=" * 80)
        logger.info(f"✅ Knowledge Base Documents Found: {len(kb_results)}")
        logger.info(f"✅ Context Enrichment: Enabled")
        logger.info(f"✅ Response Generated: Yes")
        logger.info(f"✅ Customer Email: Ready to send\n")

        logger.info("This demonstrates how FAISS KB improves response quality by:")
        logger.info("  • Finding relevant documentation automatically")
        logger.info("  • Providing context to the LLM")
        logger.info("  • Ensuring consistent, accurate answers")
        logger.info("  • Reducing hallucinations from the LLM")

    except Exception as e:
        logger.error(f"Example failed: {e}", exc_info=True)
        sys.exit(1)


async def another_example():
    """Another example: Billing inquiry."""
    try:
        logger.info("\n\n" + "=" * 80)
        logger.info("Another Example: Billing Inquiry")
        logger.info("=" * 80 + "\n")

        vector_kb = VectorKBService()
        await vector_kb.initialize()

        # Customer inquiry
        inquiry = {
            "subject": "Subscription pricing and plans",
            "body": "What are the different pricing tiers available? Do you offer annual discounts?",
        }

        logger.info(f"Customer Question: {inquiry['subject']}\n")

        # Search KB
        results = await vector_kb.search(
            query=f"{inquiry['subject']} {inquiry['body']}",
            category="billing",
            limit=3
        )

        logger.info(f"📚 Found {len(results)} relevant billing documents:\n")

        for i, doc in enumerate(results, 1):
            logger.info(f"{i}. {doc['title']}")
            logger.info(f"   Relevance: {doc['similarity_score']:.1%}")
            logger.info(f"   Source: {doc.get('source_url', 'N/A')}\n")

        # Format context
        context = await vector_kb.format_context(results)
        logger.info("Formatted context ready for LLM:\n")
        logger.info(context)

    except Exception as e:
        logger.error(f"Example failed: {e}", exc_info=True)


async def main():
    """Run all examples."""
    await example_workflow()
    await another_example()


if __name__ == "__main__":
    asyncio.run(main())
