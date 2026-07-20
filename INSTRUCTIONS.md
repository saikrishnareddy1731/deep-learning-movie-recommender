# Deep-Learning Movie Recommender

## Files

- deep-learning-movie-recommender.ipynb
- app_deep_learning.py
- requirements_deep_learning.txt

## Steps

1. Put both TMDB CSV files beside the notebook.
2. Install dependencies:

   python -m pip install -r requirements_deep_learning.txt

3. Run every notebook cell.
4. Confirm these files were generated:

   movie_dict.pkl
   movie_embeddings.npy

5. Keep the generated files beside app_deep_learning.py.
6. Run:

   python -m streamlit run app_deep_learning.py

The first Sentence Transformer run downloads the pretrained model.
Regenerate the exposed TMDB API key before publishing the project.
