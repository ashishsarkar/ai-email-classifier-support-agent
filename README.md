# Customer Support Email Agent

A LangGraph-based intelligent customer support email agent built with FastAPI, LangChain, and OpenAI. This agent automatically processes incoming support emails, understands context, and generates appropriate responses.

## Features

- **Intelligent Email Processing**: Automatically reads and categorizes incoming support emails
- **LangGraph Workflow**: State-based email processing pipeline with LangGraph
- **LLM-Powered Responses**: Uses OpenAI's GPT models to generate contextual responses
- **FastAPI Integration**: RESTful API for manual triggers and status monitoring
- **Email Integration**: IMAP/SMTP support for email reading and sending
- **Async Processing**: Non-blocking email operations for high throughput
- **Logging & Monitoring**: Comprehensive logging with Loguru

## Project Structure

```
.
├── src/
│   ├── api/                 # FastAPI routes and endpoints
│   ├── graph/              # LangGraph workflow definitions
│   ├── nodes/              # Individual graph node implementations
│   ├── services/           # Business logic services (email, llm, etc.)
│   ├── prompts/            # LLM prompt templates
│   ├── schemas/            # Pydantic models and data schemas
│   ├── core/               # Core configuration and settings
│   └── utils/              # Utility functions and helpers
├── data/                   # Knowledge base and vector stores
├── tests/                  # Unit and integration tests
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables
├── main.py                 # Application entry point
└── README.md              # This file
```

## Prerequisites

- Python 3.12+
- pip or poetry
- OpenAI API key
- Gmail account (or other IMAP/SMTP email provider)

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd customer-support-email-agent
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env  # Or configure .env directly
   ```

   Update `.env` with your:
   - OpenAI API key
   - Email credentials
   - Database URL
   - Other configuration settings

## Configuration

### Environment Variables

Key environment variables (see `.env` for complete list):

- `OPENAI_API_KEY`: Your OpenAI API key
- `OPENAI_MODEL`: Model to use (e.g., gpt-4-turbo-preview)
- `EMAIL_ADDRESS`: Email account for processing
- `EMAIL_PASSWORD`: Email account password or app-specific password
- `DATABASE_URL`: Database connection string
- `EMAIL_CHECK_INTERVAL`: Frequency of email checks in seconds

## Running the Application

### Start the API Server

```bash
python main.py
```

The API will be available at `http://localhost:8000`

### API Documentation

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Usage

### Example API Endpoints

```bash
# Process pending emails
curl -X POST http://localhost:8000/api/emails/process

# Get email statistics
curl http://localhost:8000/api/emails/stats

# Manual trigger graph execution
curl -X POST http://localhost:8000/api/graph/trigger
```

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Code Formatting

```bash
black src/ tests/
ruff check src/ tests/
```

### Type Checking

```bash
mypy src/
```

## Architecture

### LangGraph Workflow

The email processing follows a state-based workflow:

1. **Email Retrieval**: Fetch emails from IMAP server
2. **Email Parsing**: Extract and structure email content
3. **Classification**: Categorize email type and urgency
4. **Context Understanding**: Analyze customer history and context
5. **Response Generation**: Generate appropriate response using LLM
6. **Response Sending**: Send response via SMTP
7. **Logging & Storage**: Record interaction in database

### Key Components

- **Nodes**: Individual processing steps in the graph
- **Services**: Reusable business logic (EmailService, LLMService, etc.)
- **Schemas**: Pydantic models for data validation
- **Prompts**: LLM prompt templates for consistency

## Database

The application uses SQLAlchemy for ORM. Default SQLite database can be switched to PostgreSQL.

### Database Migrations

```bash
alembic upgrade head
```

## Logging

Logs are configured with Loguru and output to:
- Console (colored)
- File: `logs/app.log`

Configure log level via `LOG_LEVEL` environment variable.

## Performance Considerations

- Async/await for non-blocking operations
- Batch email processing for efficiency
- LLM caching to reduce API calls
- Connection pooling for database

## Troubleshooting

### Common Issues

1. **OpenAI API Key Not Found**
   - Ensure `OPENAI_API_KEY` is set in `.env`

2. **Email Connection Failed**
   - Verify email credentials in `.env`
   - Check firewall/network settings
   - For Gmail, use app-specific passwords

3. **LangGraph Execution Errors**
   - Check node implementations
   - Verify input schemas match expected formats
   - Check logs for detailed error messages

## Contributing

Contributions are welcome! Please follow the existing code style and add tests for new features.

## License

MIT License - See LICENSE file for details

## Support

For issues and questions, please open an issue on the repository.

## Roadmap

- [ ] Multi-language support
- [ ] Advanced email classification
- [ ] Customer sentiment analysis
- [ ] Escalation workflow
- [ ] Knowledge base integration
- [ ] Metrics and analytics dashboard
