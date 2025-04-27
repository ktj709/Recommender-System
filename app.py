import streamlit as st
import pickle


# ------------------------------
# Load trained model and data
# ------------------------------
model = pickle.load(open("model.pkl", "rb"))
book_pivot = pickle.load(open("book_pivot.pkl", "rb"))

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Book Recommender", layout="wide")
st.title("📚 Book Recommender System")
st.write("Discover books similar to the ones you love!")

# Dropdown to select a book
selected_book = st.selectbox("Choose a book:", book_pivot.index)

# Button to trigger recommendations
if st.button("🔍 Recommend"):
    # Get neighbors (including itself at index 0)
    distances, suggestions = model.kneighbors(book_pivot.loc[selected_book].values.reshape(1, -1), n_neighbors=6)

    st.subheader("📖 Recommended Books:")
    for i in range(1, 6):  # skip the book itself
        st.write(f"{i}. {book_pivot.index[suggestions[0][i]]}")
