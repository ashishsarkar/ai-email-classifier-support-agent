"""Script to populate FAISS knowledge base with sample documents."""

import asyncio
import sys

# Add parent directory to path
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

from src.services.vector_kb_service import VectorKBService
from loguru import logger


# Sample knowledge base documents
SAMPLE_DOCUMENTS = [
    # Product Information
    {
        "title": "Product Overview - CloudSync Pro",
        "content": "CloudSync Pro is our flagship cloud synchronization platform that allows teams to collaborate seamlessly. It supports real-time sync across all devices, unlimited storage, and enterprise-grade security with end-to-end encryption. Perfect for teams of any size.",
        "category": "product_inquiry",
        "source_url": "https://docs.cloudsync.com/overview",
    },
    {
        "title": "CloudSync Pro Features",
        "content": "Features include: Real-time file synchronization, Version control with 30-day history, Team collaboration tools, Mobile app support (iOS/Android), API access, SSO integration, Advanced permissions, and 99.9% uptime SLA.",
        "category": "product_inquiry",
        "source_url": "https://docs.cloudsync.com/features",
    },
    {
        "title": "CloudSync Pricing Plans",
        "content": "Free Plan: Up to 5GB storage, 1 user. Pro Plan: $9.99/month, 1TB storage, 5 users. Business Plan: $19.99/month, unlimited storage, 50 users. Enterprise: Custom pricing, dedicated support.",
        "category": "billing",
        "source_url": "https://cloudsync.com/pricing",
    },
    # Technical Support
    {
        "title": "Troubleshooting Sync Issues",
        "content": "If files aren't syncing: 1) Check internet connection, 2) Restart the app, 3) Check folder permissions, 4) Verify file isn't locked by another app, 5) Clear app cache and reinstall if needed.",
        "category": "technical_support",
        "source_url": "https://docs.cloudsync.com/troubleshoot-sync",
    },
    {
        "title": "Installation and Setup Guide",
        "content": "Download CloudSync from our website or app store. Run the installer and follow the setup wizard. Sign in with your account. Select folders to sync. The app will run in the background automatically. Check notification area for sync status.",
        "category": "technical_support",
        "source_url": "https://docs.cloudsync.com/installation",
    },
    {
        "title": "Network Connection Requirements",
        "content": "CloudSync requires a stable internet connection with minimum 1 Mbps bandwidth. Works over WiFi or cellular. Automatically detects network changes and reconnects. For offline work, files remain available locally and sync when connection is restored.",
        "category": "technical_support",
        "source_url": "https://docs.cloudsync.com/network-requirements",
    },
    {
        "title": "Mobile App Troubleshooting",
        "content": "For iOS/Android issues: Ensure app is updated to latest version, check available storage space, clear app cache from settings, disable and re-enable background sync, reinstall if problems persist.",
        "category": "technical_support",
        "source_url": "https://docs.cloudsync.com/mobile-troubleshoot",
    },
    # Billing and Accounts
    {
        "title": "Billing FAQ - Common Questions",
        "content": "Billing is monthly, charged on the same day each month. You can upgrade/downgrade anytime. Downgrades take effect next billing cycle. Unused storage doesn't roll over. All plans include 30-day free trial.",
        "category": "billing",
        "source_url": "https://docs.cloudsync.com/billing-faq",
    },
    {
        "title": "Payment Methods and Refunds",
        "content": "We accept major credit cards, PayPal, and bank transfers. Refunds are available within 30 days of purchase for annual plans. Monthly subscriptions can be cancelled anytime. No hidden fees or long-term contracts required.",
        "category": "billing",
        "source_url": "https://docs.cloudsync.com/payment-refunds",
    },
    {
        "title": "How to Update Billing Information",
        "content": "Go to Settings > Billing to update payment method. You can change card, enable auto-renew, or switch billing email. Changes take effect within 24 hours. You'll receive confirmation email for all billing changes.",
        "category": "billing",
        "source_url": "https://docs.cloudsync.com/update-billing",
    },
    # Account Management
    {
        "title": "Account Security Best Practices",
        "content": "Enable two-factor authentication in security settings. Use strong, unique passwords. Never share account credentials. Review active sessions regularly. Logout from unused devices. Update contact info for account recovery.",
        "category": "product_inquiry",
        "source_url": "https://docs.cloudsync.com/security-practices",
    },
    {
        "title": "Password Reset and Account Recovery",
        "content": "Forgot password? Click 'Forgot Password' on login. Check your email for reset link (valid 24 hours). Create new password. Lost access to email? Contact support with verification details. Recovery can take 24-48 hours.",
        "category": "technical_support",
        "source_url": "https://docs.cloudsync.com/password-reset",
    },
    {
        "title": "Deleting Your CloudSync Account",
        "content": "To delete account: Go to Settings > Account > Delete Account. Confirm deletion. Your data will be permanently deleted within 30 days. Subscription will be cancelled. Download your data before deletion if needed.",
        "category": "product_inquiry",
        "source_url": "https://docs.cloudsync.com/delete-account",
    },
    # Complaints and Issues
    {
        "title": "Service Reliability and SLA",
        "content": "CloudSync guarantees 99.9% uptime. If we experience outages, affected users receive service credits. Check status.cloudsync.com for real-time system status. We maintain multiple data centers for redundancy.",
        "category": "complaint",
        "source_url": "https://docs.cloudsync.com/sla",
    },
    {
        "title": "Data Privacy and Compliance",
        "content": "CloudSync complies with GDPR, CCPA, and HIPAA. All data encrypted end-to-end. Regular security audits. SOC 2 certified. No data sharing with third parties. Complete privacy policy available on website.",
        "category": "product_inquiry",
        "source_url": "https://cloudsync.com/privacy",
    },
    {
        "title": "Managing Team Members and Permissions",
        "content": "Invite team members via Settings > Team. Set permissions for each user: Viewer (read-only), Editor (read/write), or Admin (full access). Remove members anytime. Track all user activity in audit logs.",
        "category": "product_inquiry",
        "source_url": "https://docs.cloudsync.com/team-management",
    },
    {
        "title": "API Documentation and Integration",
        "content": "CloudSync API available for Business/Enterprise plans. Supports file operations, sharing, and metadata. Full REST API documentation. Rate limits: 1000 requests/hour. Authentication via OAuth 2.0 or API keys.",
        "category": "technical_support",
        "source_url": "https://api.cloudsync.com/docs",
    },
    {
        "title": "Large File Upload and Handling",
        "content": "CloudSync supports files up to 20GB. Use chunked upload for files >1GB. Resume uploads if interrupted. Progress indicator shows upload status. Large file uploads may take time depending on connection speed.",
        "category": "technical_support",
        "source_url": "https://docs.cloudsync.com/large-files",
    },
    {
        "title": "Export and Data Portability",
        "content": "Export your data anytime via Settings > Data Export. Available in ZIP format with all files and metadata. Takes 24-48 hours for large accounts. Download link valid for 30 days. Full data portability guaranteed.",
        "category": "product_inquiry",
        "source_url": "https://docs.cloudsync.com/data-export",
    },
    {
        "title": "Enterprise Features and Dedicated Support",
        "content": "Enterprise plans include: Dedicated account manager, 24/7 priority support, Custom SLA, Advanced analytics, White-label options, Dedicated servers, Custom integrations, Compliance assistance.",
        "category": "product_inquiry",
        "source_url": "https://cloudsync.com/enterprise",
    },
]


async def populate_knowledge_base():
    """Populate FAISS knowledge base with sample documents."""
    try:
        logger.info("Initializing Vector Knowledge Base Service...")
        kb_service = VectorKBService()
        await kb_service.initialize()

        logger.info(f"Adding {len(SAMPLE_DOCUMENTS)} documents to knowledge base...")

        for doc in SAMPLE_DOCUMENTS:
            doc_id = await kb_service.add_document(
                title=doc["title"],
                content=doc["content"],
                category=doc["category"],
                source_url=doc["source_url"],
            )
            logger.info(f"Added document {doc_id}: {doc['title']}")

        # Test searches
        logger.info("\n" + "=" * 80)
        logger.info("Testing Knowledge Base Searches")
        logger.info("=" * 80 + "\n")

        test_queries = [
            ("How do I fix sync issues?", "technical_support"),
            ("What are the pricing plans?", "billing"),
            ("How do I reset my password?", None),
            ("Tell me about enterprise features", "product_inquiry"),
            ("Payment methods accepted?", "billing"),
        ]

        for query, category in test_queries:
            logger.info(f"🔍 Query: {query}")
            if category:
                logger.info(f"   Category filter: {category}")

            results = await kb_service.search(query, category=category, limit=3)

            if results:
                for i, result in enumerate(results, 1):
                    logger.info(f"   {i}. {result['title']} (Relevance: {result['similarity_score']:.1%})")
            else:
                logger.info("   No results found")

            logger.info("")

        # Display summary
        health = await kb_service.health_check()
        logger.info("=" * 80)
        logger.info("Knowledge Base Summary")
        logger.info("=" * 80)
        logger.info(f"Status: {health['status']}")
        logger.info(f"Total Documents: {health['documents_count']}")
        logger.info(f"Embedding Model: {health['embedding_model']}")
        logger.info(f"Vector Dimension: {health['index_dimension']}")

        logger.info("\n✅ Knowledge base population complete!")

    except Exception as e:
        logger.error(f"Failed to populate knowledge base: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(populate_knowledge_base())
