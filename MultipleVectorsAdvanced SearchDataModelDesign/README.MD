#### Utilizing Multiple Vectors and Advanced Search Data Model Design

# Goal of this Application

In this application, we will build an advanced data model and use it for ingest and various search options.   For this notebook portion, we will

### 1️⃣ Ingest Data Fields, Enrich Data With Lookups, and Format :
Learn to ingest data from including JSON and Images, format and transform to optimize hybrid searches.   This is done inside the streetcams.py application.

### 2️⃣ Store Data into Milvus:
Learn to store data into Milvus, an efficient vector database designed for high-speed similarity searches and AI applications.   In this step we are optimizing data model with scalar and multiple vector fields -- one for text and one for the camera image.   We do this in the streetcams.py application.

### 3️⃣ Use Open Source Models for Data Queries in a Hybrid Multi-Modal, Multi-Vector Search:
Discover how to use scalars and multiple vectors to query data stored in Milvus and re-rank the final results in this notebook.

### 4️⃣ Display resulting text and images:
Build a quick output for validation and checking in this notebook.

### 5️⃣ Simple Retrieval-Augmented Generation (RAG) with LangChain:
Build a simple Python [RAG](https://milvus.io/docs/integrate_with_langchain.md) application (streetcamrag.py) to use Milvus for asking about the current weather via OLLAMA.   While outputing to the screen we also send the results to Slack formatted as Markdown.

# 🔍 Summary
By the end of this application, you’ll have a comprehensive understanding of using Milvus, data ingest object semi-structured and unstructured data, and using Open Source models to build a robust and efficient data retrieval system.   For future enhancements, we can use these results to build prompts for LLM, slack bots, streaming data to Kafka and as a Street Camera search engine.


![diagram](https://github.com/tspannhw/AIM-NYCStreetCams/blob/main/MultipleVectorsAdvanced%20SearchDataModelDesign/Untitled-2024-06-10-1526.png?raw=true)


### Future Features

* https://github.com/tspannhw/AIM-AirQuality/tree/main


### Resources

* https://511ny.org/developers/help/api/get-api-getcameras_key_format






