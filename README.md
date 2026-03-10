Recommend Gym Exercises

A content-based recommendation system for gym exercises using BERT embeddings and injury-aware filtering.

Overview
This project recommends gym exercises based on user input using multiple recommendation modes:
- description: semantic search using BERT embeddings on exercise descriptions
- title: similarity search on exercise titles
- bodypart: filter exercises by body part
- goal: recommend exercises based on workout goal templates

The system supports filtering by:
- workout type
- equipment
- difficulty level
- injury constraints (spine, shoulder, knee, wrist, neck)

Dataset
The dataset contains 1,050 exercises with the following columns:
- Title
- Desc
- Type
- BodyPart
- Equipment
- Level
- Rating
- Bert_Embedding (768-d vector generated from Desc)

Data files:
- data/data.csv: raw dataset
- data/data_with_bert.pkl: dataset enriched with BERT embeddings

How It Works

1) BERT Embeddings
We use a pre-trained BERT model to convert each exercise description into a dense vector representation (embedding).
Embeddings are generated using the [CLS] pooled output from BERT (768 dimensions).

2) Similarity Search
When a user enters a query, it is embedded using the same BERT model.
Cosine similarity is computed between the query embedding and all exercise embeddings.
Top-K most similar exercises are returned.

3) Recommendation Modes

Mode: description
Input: free-text query (e.g., "strengthen lower back safely")
Output: top exercises with semantically similar descriptions

Mode: title
Input: exercise name (e.g., "push up")
Output: similar exercise titles

Mode: bodypart
Input: body part (e.g., "chest")
Output: exercises targeting the selected body part

Mode: goal
Input: workout goal (e.g., "fat loss", "muscle gain")
Output: exercises matching templates for that goal

4) Injury-Aware Filtering
If injury filters are enabled, the system excludes exercises that may aggravate certain injuries:
- spine: avoids heavy squats, deadlifts, spinal loading
- shoulder: avoids overhead presses, heavy bench variations
- knee: avoids deep squats, high-impact movements
- wrist: avoids push-up variations stressing the wrists
- neck: avoids heavy shrugs, neck loading movements

Installation

1) Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  (macOS/Linux)
venv\Scripts\activate     (Windows)

2) Install dependencies
pip install -r requirements.txt

Usage

Quickstart
Run the demo script:
python quickstart.py

Example: Description Mode
from src.recommender import recommend_exercises

results = recommend_exercises(
    query="strengthen lower back safely",
    mode="description",
    top_k=10,
    injury="spine",
    equipment=None,
    level=None,
    workout_type=None
)

print(results)

Example: Title Mode
results = recommend_exercises(
    query="push up",
    mode="title",
    top_k=10
)

Example: BodyPart Mode
results = recommend_exercises(
    query="chest",
    mode="bodypart",
    top_k=20
)

Example: Goal Mode
results = recommend_exercises(
    query="fat loss",
    mode="goal",
    top_k=15
)

Project Structure
.
## Project Structure
.
├── app.py
├── quickstart.py
├── requirements.txt
├── data/
│   ├── data.csv
│   └── data_with_bert.pkl
├── model_bert/
│   ├── config.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── model.safetensors
└── src/
    ├── recommender.py
    └── utils.py
Notes
- The description mode requires precomputed embeddings in data/data_with_bert.pkl.
- If you modify the dataset, you must regenerate BERT embeddings.
- For best performance, keep embeddings cached and avoid recomputing for every query.

Troubleshooting

1) FileNotFoundError for data_with_bert.pkl
Make sure you have generated embeddings or included the file in the correct path:
data/data_with_bert.pkl

2) Model loading issues
Ensure the BERT model files exist in model_bert/ or update paths in src/recommender.py.

3) Slow inference
Embedding generation can be slow. Consider:
- using smaller transformer model
- caching query embeddings
- using faiss for faster nearest neighbor search

Future Improvements
- Add FAISS-based retrieval for faster similarity search
- Add hybrid recommendation (metadata + embeddings)
- Improve injury filtering with rule + learned classifier
- Deploy Streamlit app with Docker