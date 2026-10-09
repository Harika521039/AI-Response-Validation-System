# AI Response Validation System with Hallucination Detection Assistance

**Live Demo:** [Open the AI Response Validation System](https://ai-response-validation-system-harika.streamlit.app/)

A multi-agent AI validation framework designed to evaluate AI-generated responses for accuracy, relevance, completeness, consistency, and potential hallucinations using Retrieval-Augmented Generation (RAG), evidence-based evaluation, and intelligent judge agents.

## Overview

The AI Response Validation System helps assess whether AI-generated answers are supported by reference evidence. It retrieves relevant information from a knowledge base and evaluates responses using multiple validation components.

The project integrates four milestones, covering retrieval, judge agents, completeness and verdict assessment, batch evaluation, analytics, and PDF report generation.

## Key Features

| Feature                   | Description                                                                 |
| ------------------------- | --------------------------------------------------------------------------- |
| Hallucination Detection   | Identifies potentially unsupported or incorrect statements in AI responses. |
| RAG-Based Retrieval       | Retrieves relevant reference evidence from a knowledge base.                |
| Multi-Agent Evaluation    | Uses multiple judge components to assess AI-generated responses.            |
| Accuracy Assessment       | Evaluates whether the response agrees with available evidence.              |
| Relevance Assessment      | Checks how well the response addresses the question.                        |
| Completeness Assessment   | Evaluates whether important information is included.                        |
| Verdict Generation        | Combines evaluation results to produce an overall assessment.               |
| Single Evaluation         | Evaluates one AI-generated response at a time.                              |
| Batch Evaluation          | Processes multiple responses in a batch.                                    |
| Evaluation History        | Stores and displays previous evaluation results.                            |
| Analytics Dashboard       | Displays evaluation statistics and performance information.                 |
| PDF Report Export         | Generates downloadable evaluation reports.                                  |
| Knowledge Base Management | Supports retrieval-based evidence grounding.                                |
| Testing and System Status | Provides access to testing information and system status.                   |

## System Architecture

The system follows a retrieval and evaluation workflow.

```text
             User Input
                 |
                 v
        Streamlit Web Interface
                 |
                 v
       Input and Response Processing
                 |
                 v
       RAG-Based Evidence Retrieval
                 |
                 v
          Vector Database
                 |
                 v
          Judge Components
                 |
        +--------+--------+
        |        |        |
        v        v        v
     Accuracy Relevance Completeness
        |        |        |
        +--------+--------+
                 |
                 v
          Verdict Generation
                 |
                 v
        Evaluation Results
                 |
          +------+------+
          |             |
          v             v
      Analytics      PDF Reports
```

## Project Milestones

### Milestone 1: RAG Retrieval Foundation

* Establishes the retrieval pipeline.
* Retrieves relevant evidence from the knowledge base.
* Provides reference context for evaluating AI-generated responses.

### Milestone 2: Judge Agents and Consistency Validation

* Introduces judge components for response assessment.
* Evaluates the consistency of AI-generated responses.
* Supports evidence-based quality assessment.

### Milestone 3: Completeness Judge, Verdict Agent, and Batch Evaluation

* Adds completeness assessment.
* Combines evaluation results into an overall verdict.
* Supports resilient batch evaluation.

### Milestone 4: Analytics Dashboard and PDF Report Export

* Provides evaluation analytics and statistics.
* Supports evaluation history and performance tracking.
* Generates PDF reports for evaluation results.

## Technology Stack

| Technology            | Purpose                                     |
| --------------------- | ------------------------------------------- |
| Python                | Main programming language                   |
| Streamlit             | Web interface                               |
| ChromaDB              | Vector storage and retrieval                |
| Sentence Transformers | Text embeddings and semantic representation |
| RAG                   | Evidence retrieval and grounding            |
| Judge Agents          | Response quality assessment                 |
| Pytest                | Automated testing                           |
| ReportLab             | PDF report generation                       |

## Application Modules

The application provides the following sections:

* **Dashboard:** View overall evaluation information.
* **Single Evaluation:** Assess an individual AI response.
* **Batch Evaluation:** Evaluate multiple responses.
* **Evaluation History:** Review previous evaluations.
* **Knowledge Base:** Manage or inspect reference information.
* **Testing / System Status:** Review system readiness and testing information.
* **About Project:** View project details.

## Getting Started

### Prerequisites

* Python 3.10 or a compatible supported Python version
* pip
* Git
* Visual Studio Code (recommended)

### 1. Clone the Repository



```bash
git clone https://github.com/Harika521039/AI-Response-Validation-System.git
cd AI-Response-Validation-System
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app.py
```

The application will normally open in your browser at:

`http://localhost:8501`

## Running Tests

Run the automated test suite from the project root:

```bash
python -m pytest tests/ -v
```

**Latest test results:** 332 passed, 5 skipped, 0 failed.

The test suite completed in approximately 58 seconds on the local development environment.


## Evaluation Workflow

1. Enter an AI-generated response and the relevant question.
2. Retrieve supporting evidence from the knowledge base.
3. Evaluate the response using the available judge components.
4. Review the evaluation scores and final verdict.
5. Save or review the evaluation in the history section.
6. Export a PDF report when required.
7. Use the analytics dashboard to review evaluation performance.

## Deployment

The project is deployed using Streamlit Community Cloud.

**Live Application:** [AI Response Validation System](https://ai-response-validation-system-harika.streamlit.app/)

To deploy an updated version, push the latest project changes to the connected GitHub repository and check the deployment status in Streamlit Community Cloud.

## Future Enhancements

* Improve hallucination detection using more advanced evaluation methods.
* Expand the reference knowledge base.
* Add more evaluation metrics and benchmarking datasets.
* Improve the visualization of evaluation trends.
* Enhance evaluation speed and scalability.

## Author

**Harika Yarakaraju**

B.Tech — Computer Science and Engineering (AI & Data Science)

Shri Vishnu Engineering College for Women

GitHub: [Harika521039](https://github.com/Harika521039)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

