import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# ---------------- Load Model & Vectorizer ----------------
with open(r"C:\Users\singh\OneDrive\Desktop\reviews_detection_1\best_model .pkl", "rb") as f:
    model = pickle.load(f)

with open(r"C:\Users\singh\OneDrive\Desktop\reviews_detection_1\best_vectorizer .pkl", "rb") as f:
    vectorizer = pickle.load(f)





# Page background styling
page_bg_img = """
<style>
.stApp {
    background-image: url("https://www.rand.org/content/rand/pubs/perspectives/PEA2679-1/jcr:content/par/teaser.crop.1200x900.ct.jpeg/1694029165260.jpeg ");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)
# ---------------- Streamlit Page Config ----------------
st.set_page_config(page_title="Review Rating Detection", layout="centered")
st.title("Review/ Feedback  Rating Detection ")
st.write("Ratings: 1 = Poor | 2 = Moderate | 3 = Good | 4 = Very Good | 5 = Excellent")

# ---------------- Rating Map ----------------
rating_map = {
    1: "Poor",
    2: "Moderate",
    3: "Good",
    4: "Very Good",
    5: "Excellent"
}

# ---------------- Single Review Prediction ----------------
st.subheader("Predict Single Review")
review = st.text_area("Enter your review here")

if st.button("Predict Single Review"):
    if review.strip() == "":
        st.warning("Please enter a review")
    else:
        review_vec = vectorizer.transform([review])
        prediction = model.predict(review_vec)[0]
        st.success(f" Predicted Rating: {prediction}")
        st.info(f"Feedback: {rating_map[prediction]}")

# ---------------- File Upload Prediction ----------------
st.subheader("Upload CSV / Excel File")
uploaded_file = st.file_uploader(
    "Upload file (must contain a column named 'review')",
    type=["csv", "xlsx"]
)

if uploaded_file:
    try:
        # Read CSV or Excel
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            try:
                df = pd.read_excel(uploaded_file)
            except ImportError:
                st.error("'openpyxl' is required to read Excel files. Install it via 'pip install openpyxl'")
                st.stop()

        # Check 'review' column
        if "review" not in df.columns:
            st.error(" File must contain a column named 'review'")
        else:
            # Predict for all reviews
            review_vectors = vectorizer.transform(df["review"].astype(str))
            df["Predicted_Rating"] = model.predict(review_vectors)
            df["Sentiment"] = df["Predicted_Rating"].map(rating_map)

            st.success(" Predictions completed!")

            # Display DataFrame
            st.subheader(" Predicted Results")
            st.dataframe(df)

            # Sentiment counts
            sentiment_counts = df["Sentiment"].value_counts()

            # Pie chart
            st.subheader(" Sentiment Distribution")
            fig, ax = plt.subplots()
            ax.pie(
                sentiment_counts,
                labels=sentiment_counts.index,
                autopct="%1.1f%%",
                startangle=45
            )
            ax.axis("equal")
            st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Error reading file: {e}")

# ---------------- About ----------------