import streamlit as st
import pickle
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="BookMatch",
    page_icon="📚",
    layout="centered"
)

# Load Models
model = pickle.load(open('notebook/artifacts/model.pkl', 'rb'))
book_pivot = pickle.load(open('notebook/artifacts/book_pivot.pkl', 'rb'))

# Recommendation Function
def recommend_book(book_name, num_recommendations):

    try:
        book_id = np.where(book_pivot.index == book_name)[0][0]

        distances, suggestions = model.kneighbors(
            book_pivot.iloc[book_id, :].values.reshape(1, -1),
            n_neighbors=num_recommendations + 1
        )

        recommended_books = []

        for i in suggestions[0][1:]:
            recommended_books.append(book_pivot.index[i])

        return recommended_books

    except Exception as e:
        return [f"Error: {e}"]


# Sidebar
st.sidebar.title("📖 About Project")

st.sidebar.info(
    """
    This Book Recommendation System uses:

    • Collaborative Filtering

    • K-Nearest Neighbors (KNN)

    • Cosine Similarity

    • Streamlit

    • Scikit-Learn

    Built as a Machine Learning Portfolio Project.
    """
)

# Main Title
st.title("📚 BookMatch AI")
st.subheader("Find books similar to your favorites")

st.write(
    "Select a book and discover recommendations generated using Machine Learning."
)

# Statistics
st.metric("Books Available", len(book_pivot.index))

st.markdown("---")

# Book Selection
selected_book = st.selectbox(
    "Choose a Book",
    book_pivot.index
)



# Button
if st.button("🔍 Recommend Books"):

    recommendations = recommend_book(
        selected_book,
        num_recommendations
    )

    st.success(f"Top {num_recommendations} Recommendations")

    for idx, book in enumerate(recommendations, start=1):
        st.write(f"**{idx}. {book}**")

