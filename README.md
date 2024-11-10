# Code Plagiarism Detection System
## Technical Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Models](#models)
4. [Plagiarism Detection Methods](#plagiarism-detection-methods)
5. [API Endpoints](#api-endpoints)
6. [Storage and File Management](#storage-and-file-management)
7. [Performance Metrics](#performance-metrics)

## System Overview
The system is designed to detect code plagiarism using a hybrid approach that combines traditional n-gram based similarity detection with advanced neural language models. It's built as a FastAPI application that integrates with Databricks for storage and MLflow for experiment tracking.

## Architecture
The system follows a microservices architecture with the following components:
- FastAPI backend for REST API endpoints
- Databricks File System (DBFS) for file storage
- MLflow for experiment tracking and model versioning
- Multiple transformer-based models for code similarity detection
- Background task processing for asynchronous plagiarism checks

## Models

### 1. Traditional N-gram Based Detection
- Uses a sliding window of 5 tokens to generate n-grams from code
- Calculates similarity based on the overlap of n-grams between submissions
- Threshold: Submissions with > 60% n-gram similarity are flagged
- Advantages: Fast, interpretable, good at catching direct code copying

### 2. Neural Language Models
The system employs two specialized models for code similarity detection:

#### CodeBERT Model
- Purpose: Primary code similarity detection
- Architecture: BERT-based transformer model pretrained on code
- Training Data: Trained on both programming language and natural language
- Advantages:
  - Understands code semantics
  - Can detect plagiarism even with variable renaming
  - Better at understanding code structure

#### GraphCodeBERT Model
- Purpose: Secondary code similarity detection with focus on code structure
- Architecture: Enhanced BERT model that incorporates code structure
- Key Features:
  - Understands data flow graphs
  - Better at detecting structural similarities
  - Can identify similar algorithms even with different implementations

### Embedding Process
The system uses a sophisticated embedding approach:

1. **Embedding Generation**
   - Code is tokenized using model-specific tokenizers
   - Maximum sequence length: 512 tokens
   - Embeddings are generated from the last hidden state
   - Mean pooling is applied to get a fixed-size representation

2. **Similarity Calculation**
   ```python
   def cosine_similarity(vec1: torch.Tensor, vec2: torch.Tensor) -> float:
       return torch.nn.functional.cosine_similarity(vec1, vec2, dim=1).item()
   ```

3. **Multiple Model Approach**
   - Uses both CodeBERT and GraphCodeBERT embeddings
   - Allows for different perspectives on code similarity
   - Reduces false positives through model consensus

## Plagiarism Detection Methods

### Combined Approach
The system uses three methods in parallel:

1. **Traditional N-gram Matching**
   - Used for exact or near-exact matches
   ```python
   def get_ngrams(code: str, n: int = 5) -> List[str]:
       tokens = code.split()
       ngrams = zip(*[tokens[i:] for i in range(n)])
       return [' '.join(ngram) for ngram in ngrams]
   ```

2. **CodeBERT Semantic Analysis**
   - Used for semantic similarity
   - Better at detecting logic similarity

3. **GraphCodeBERT Structural Analysis**
   - Used for structural similarity
   - Better at detecting algorithm similarity

### Result Processing
- Results from all three methods are combined
- Each match includes:
  - Similarity score
  - Match type (Traditional/LLM-based_code/LLM-based_graph)
  - Timestamp
  - File information

## Storage and File Management

### DBFS Integration
- Uses Databricks File System for storing:
  - Original submissions
  - Analysis results
  - Reports
- Directory structure:
  ```
  /baza/
    /{user_name}/
      - submission files
    /results/
      - analysis results
  ```

### MLflow Tracking
- Tracks:
  - Similarity scores
  - Number of matches
  - Model performance metrics
  - Analysis parameters
  - Report artifacts

## Performance Metrics
The system tracks several key metrics through MLflow:
- Traditional match count
- LLM-based match count (both models)
- Average similarity scores
- Processing time
- False positive rates (through manual verification)

## API Endpoints

### Main Endpoints
1. `/plagiarism/upload-and-check/`
   - Uploads and initiates plagiarism check
   - Supports async processing

2. `/plagiarism/report/{submission_id}`
   - Generates detailed reports
   - Supports HTML and JSON formats

3. `/plagiarism/reports/summary`
   - Provides summary of all checks
   - Supports date range filtering

### Report Generation
- Generates both machine-readable (JSON) and human-readable (HTML) reports
- Includes:
  - Similarity scores
  - Match details
  - Visualization data
  - Temporal analysis

## Security Considerations
- Authentication using Bearer tokens
- File access controls through Databricks
- Secure file handling with temporary storage
- Input validation and sanitization
- Rate limiting on API endpoints

## Usage Recommendations
1. Use multiple similarity thresholds based on context
2. Regular model updates and fine-tuning
3. Monitor false positive/negative rates
4. Implement manual review process for high-confidence matches
5. Regular backup of submission database

This system represents a comprehensive approach to code plagiarism detection, combining traditional methods with state-of-the-art AI models to provide accurate and reliable results.
