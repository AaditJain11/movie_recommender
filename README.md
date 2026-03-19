# 🎬 Movie Recommendation System

This project is a **Content-Based Movie Recommendation System** that suggests movies similar to a selected title using **machine learning techniques like TF-IDF and Cosine Similarity**.

It enhances recommendation quality by applying filters such as **genre, rating, and popularity (number of votes)**, ensuring that the results are both relevant and high-quality.

---

 Features

*  Recommend movies based on content similarity
*  Uses **TF-IDF Vectorization** for better text understanding
*  Uses **Cosine Similarity** to measure movie similarity
*  Filter recommendations by:
  > Genres
  > Minimum & maximum ratings
  > Number of ratings (popularity)
*  Sorted recommendations based on similarity and rating
*  Efficient and scalable approach



## 🛠️ Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
*  Streamlit (For ui and deployment)

 How It Works

1. Data Preprocessing

   * Combine features like genres, tags, and keywords into a single column (`soup`)

2. Vectorization

   * Apply TF-IDF Vectorizer to convert text data into numerical vectors

3. Similarity Calculation

   * Compute **cosine similarity** between all movies

4. Filtering

   Apply filters based on:
     * Genre
     * Rating
     * Number of votes

5. Recommendation
   * Select top similar movies
   * Sort by similarity and rating
   * Display final recommendations



 Project Structure


├── data/                 # Dataset files
├── notebook.ipynb       # Main implementation
├── app.py               # (Optional) Streamlit app
├── README.md


 Objective
The goal of this project is to build a **real-world recommendation system** that:

* Understands movie content
* Provides personalized suggestions
* Improves recommendation quality using filtering

 Future Improvements

*  Add Streamlit Web App UI
*  Include cast & crew weighting
*  Add user-based recommendations
*  Deploy on cloud (AWS / Render)



 Acknowledgements

Dataset: MovieLens / TMDB



 License

This project is open-source and available under the MIT License.
