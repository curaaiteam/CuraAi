# 🤖 CuraAi - Empathetic AI Companion AI Backend

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> An empathetic AI companion that provides emotional support and thoughtful guidance through conversational AI.

## 🌟 Features

- 💬 **Conversational AI** - Natural, empathetic responses to user messages
- 🧠 **Memory System** - Maintains conversation context using Pinecone vector database
- 🔐 **Session Management** - Email-based user sessions for personalized interactions
- 🚀 **REST API** - Easy-to-integrate FastAPI backend
- 🌍 **CORS Enabled** - Ready for frontend integration

## ⚠️ Important Notice

This API runs on **free-tier resources** with limited compute power:
- **Initial Request**: ~90 seconds (1 min 30 sec)
- **Subsequent Requests**: ~60-75 seconds (1 min - 1 min 15 sec)

Please implement appropriate loading indicators and set user expectations accordingly.

## 📋 Table of Contents

- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## 🚀 Installation

### Prerequisites

- Python 3.9 or higher
- Pinecone account and API key
- Hugging Face account (optional, for model access)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/curaai.git
cd curaai
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

4. **Create `.env` file**
```bash
cp .env.example .env
```

5. **Configure environment variables** (see [Configuration](#configuration))

## ⚙️ Configuration

Create a `.env` file in the root directory with the following variables:

```env
# API Security
SECRET_KEY=curaai_access_key

# Pinecone Configuration
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=us-east-1

# Hugging Face (Optional)
HF_TOKEN=your_huggingface_token_here
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here

# Model Configuration
MODEL_NAME=meta-llama/Llama-3.2-3B
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

### Environment Variables Explained

| Variable | Description | Required |
|----------|-------------|----------|
| `SECRET_KEY` | API authentication key | Yes |
| `PINECONE_API_KEY` | Pinecone vector database API key | Yes |
| `PINECONE_ENVIRONMENT` | Pinecone environment region | Yes |
| `HF_TOKEN` | Hugging Face API token | Optional |
| `MODEL_NAME` | LLM model identifier | Optional |
| `EMBEDDING_MODEL` | Sentence transformer model | Optional |

## 🎯 Usage

### Running Locally

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### Running in Production

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Testing the API

```bash
curl -X POST http://localhost:8000/ai-chat \
  -H "Content-Type: application/json" \
  -H "x-api-key: curaai_access_key" \
  -d '{
    "query": "I need someone to talk to",
    "session_id": "user@example.com"
  }'
```

## 📚 API Documentation

### Base URL
```
https://curaaiteam-curaai.hf.space
```

### Endpoint: POST `/ai-chat`

Send a message and receive an empathetic AI response.

#### Request Headers
```http
Content-Type: application/json
x-api-key: curaai_access_key
```

#### Request Body
```json
{
  "query": "I'm feeling overwhelmed with work",
  "session_id": "user@example.com"
}
```

#### Response
```json
{
  "reply": "I hear you - feeling overwhelmed at work can be really draining. What's been weighing on you the most today?",
  "session_id": "user@example.com"
}
```

#### Error Responses

| Status Code | Description |
|-------------|-------------|
| 400 | Missing session_id |
| 403 | Invalid API key |
| 500 | Model failed to respond |

### Integration Examples

#### JavaScript
```javascript
const sendMessage = async (userMessage, userEmail) => {
  const response = await fetch('https://curaaiteam-curaai.hf.space/ai-chat', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'x-api-key': 'curaai_access_key'
    },
    body: JSON.stringify({
      query: userMessage,
      session_id: userEmail
    })
  });
  
  const data = await response.json();
  return data.reply;
};
```

#### Python
```python
import requests

def send_message(user_message: str, user_email: str) -> str:
    response = requests.post(
        'https://curaaiteam-curaai.hf.space/ai-chat',
        headers={
            'Content-Type': 'application/json',
            'x-api-key': 'curaai_access_key'
        },
        json={
            'query': user_message,
            'session_id': user_email
        }
    )
    return response.json()['reply']
```

For complete API documentation, see [API_DOCS.md](API_DOCS.md)

## 📁 Project Structure

```
curaai/
├── main.py              # FastAPI application and endpoints
├── vector.py            # Pinecone memory management
├── requirements.txt     # Python dependencies
├── .env.example        # Environment variables template
├── .env                # Environment variables (not in git)
├── README.md           # This file
└── API_DOCS.md         # Complete API documentation
```

### Key Files

- **`main.py`**: Contains the FastAPI application, LLM setup, and `/ai-chat` endpoint
- **`vector.py`**: Manages Pinecone vector database operations for conversation memory
- **`requirements.txt`**: Lists all Python package dependencies

## 🛠️ Technical Stack

- **Framework**: FastAPI
- **LLM**: Llama 3.2 3B (via Hugging Face)
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Vector DB**: Pinecone
- **ML Libraries**: PyTorch, Transformers, LangChain

## 🔒 Security

- API key authentication via headers
- Email-based session isolation
- No cross-session data leakage
- CORS configured for all origins (configure appropriately for production)

## 🧪 Testing

### Manual Testing
```bash
# Test endpoint availability
curl http://localhost:8000/

# Test chat endpoint
curl -X POST http://localhost:8000/ai-chat \
  -H "Content-Type: application/json" \
  -H "x-api-key: curaai_access_key" \
  -d '{"query": "Hello", "session_id": "test@example.com"}'
```

## 🐛 Troubleshooting

### Common Issues

**Issue**: `403 Forbidden`
- **Solution**: Check that `x-api-key` header matches your `SECRET_KEY` in `.env`

**Issue**: `PINECONE_API_KEY must be set`
- **Solution**: Ensure Pinecone API key is set in `.env` file

**Issue**: Model download taking too long
- **Solution**: First run downloads the model. Subsequent runs will be faster.

**Issue**: CUDA out of memory
- **Solution**: Reduce batch size or use CPU mode by setting `DEVICE=-1`

## 📖 Documentation

- [Complete API Documentation](API_DOCS.md)
- [FastAPI Auto Docs](http://localhost:8000/docs) (when running locally)
- [Pinecone Documentation](https://docs.pinecone.io/)

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guide for Python code
- Add docstrings to all functions
- Update documentation for API changes
- Test thoroughly before submitting PR

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **CuraAi Team** - *Initial work*

## 🙏 Acknowledgments

- Hugging Face for model hosting
- Pinecone for vector database
- FastAPI community
- Open source contributors

## 📞 Support

For issues, questions, or feature requests:
- Open an issue on GitHub
- Contact: support@curaai.com (if available)

## 🗺️ Roadmap

- [ ] Add user feedback mechanism
- [ ] Implement conversation export feature
- [ ] Add multi-language support
- [ ] Improve response time optimization
- [ ] Add analytics dashboard
- [ ] Implement rate limiting per user

---

**Built with ❤️ by the CuraAi Team**
