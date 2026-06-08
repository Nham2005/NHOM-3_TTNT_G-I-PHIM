import pickle
import numpy as np
import pandas as pd
import streamlit as st

from scipy.sparse import vstack
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(
    page_title="Hybrid Movie Recommendation",
    layout="wide"
)


# ===============================
# 1. LOAD MODEL VÀ DỮ LIỆU
# ===============================
@st.cache_resource
def load_artifacts():
    with open("model/svd_model.pkl", "rb") as f:
        svd_model = pickle.load(f)

    with open("model/tfidf_matrix.pkl", "rb") as f:
        tfidf_matrix = pickle.load(f)

    with open("model/movies_nlp.pkl", "rb") as f:
        movies_nlp = pickle.load(f)

    with open("model/ratings_train.pkl", "rb") as f:
        ratings_train = pickle.load(f)

    with open("model/movie_id_to_index.pkl", "rb") as f:
        movie_id_to_index = pickle.load(f)

    return svd_model, tfidf_matrix, movies_nlp, ratings_train, movie_id_to_index


svd_model, tfidf_matrix, movies_nlp, ratings_train, movie_id_to_index = load_artifacts()


# ===============================
# 2. HÀM PHÂN TÍCH THỂ LOẠI USER THÍCH
# ===============================
def get_user_favorite_genres(user_id, ratings_df, movies_df, min_rating=3.5):
    user_ratings = ratings_df[
        (ratings_df["userId"] == user_id) &
        (ratings_df["rating"] >= min_rating)
    ].copy()

    if user_ratings.empty:
        return pd.DataFrame(columns=["genre", "count", "avg_rating"])

    user_movies = user_ratings.merge(
        movies_df[["movieId", "genres"]],
        on="movieId",
        how="left"
    )

    genre_stats = {}

    for row in user_movies.itertuples():
        genres_text = str(row.genres).replace("|", " ")
        genres = genres_text.split()

        for genre in genres:
            if genre not in genre_stats:
                genre_stats[genre] = {
                    "total_rating": 0,
                    "count": 0
                }

            genre_stats[genre]["total_rating"] += row.rating
            genre_stats[genre]["count"] += 1

    result = []

    for genre, stats in genre_stats.items():
        result.append({
            "genre": genre,
            "count": stats["count"],
            "avg_rating": round(stats["total_rating"] / stats["count"], 3)
        })

    return (
        pd.DataFrame(result)
        .sort_values(by=["avg_rating", "count"], ascending=False)
        .reset_index(drop=True)
    )


# ===============================
# 3. HÀM TẠO HỒ SƠ SỞ THÍCH USER
# ===============================
def build_user_profile(user_id, ratings_df, min_rating=3.5):
    user_ratings = ratings_df[
        (ratings_df["userId"] == user_id) &
        (ratings_df["rating"] >= min_rating)
    ].copy()

    if user_ratings.empty:
        return None

    movie_vectors = []
    weights = []

    for row in user_ratings.itertuples():
        movie_id = row.movieId
        if movie_id in movie_id_to_index:
            movie_index = movie_id_to_index[movie_id]
            movie_vectors.append(tfidf_matrix[movie_index])
            weights.append(row.rating)

    if len(movie_vectors) == 0:
        return None

    movie_matrix = vstack(movie_vectors)
    weights = np.array(weights)

    user_profile = movie_matrix.multiply(weights[:, None]).sum(axis=0) / weights.sum()

    return np.asarray(user_profile)


# ===============================
# 4. HÀM TÍNH CONTENT SCORE
# ===============================
def get_user_content_scores(user_id, ratings_df, min_rating=3.5):
    user_profile = build_user_profile(
        user_id=user_id,
        ratings_df=ratings_df,
        min_rating=min_rating
    )

    if user_profile is None:
        return None

    content_scores = cosine_similarity(
        user_profile,
        tfidf_matrix
    ).flatten()

    return content_scores


# ===============================
# 5. HÀM CHUẨN HÓA ĐIỂM SVD
# ===============================
def normalize_svd_score(score, min_rating=0.5, max_rating=5.0):
    normalized_score = (score - min_rating) / (max_rating - min_rating)
    return max(0, min(1, normalized_score))


# ===============================
# 6. HÀM GỢI Ý PHIM HYBRID
# ===============================
def recommend_movies_for_user(
    user_id,
    ratings_df,
    top_n=10,
    alpha=0.7,
    min_rating_for_profile=3.5
):
    content_scores = get_user_content_scores(
        user_id=user_id,
        ratings_df=ratings_df,
        min_rating=min_rating_for_profile
    )

    if content_scores is None:
        return pd.DataFrame({
            "message": [f"User {user_id} chưa có đủ phim rating cao để tạo hồ sơ sở thích."]
        })

    watched_movie_ids = set(
        ratings_df[ratings_df["userId"] == user_id]["movieId"]
    )

    recommendations = []

    for movie_position, movie in enumerate(movies_nlp.itertuples(index=False)):
        movie_id = movie.movieId

        if movie_id in watched_movie_ids:
            continue

        svd_pred_rating = svd_model.predict(user_id, movie_id).est
        svd_score_norm = normalize_svd_score(svd_pred_rating)
        content_score = content_scores[movie_position]

        hybrid_score = alpha * svd_score_norm + (1 - alpha) * content_score

        recommendations.append({
            "movieId": movie_id,
            "title": movie.title,
            "genres": movie.genres,
            "svd_pred_rating": round(svd_pred_rating, 3),
            "svd_score_norm": round(svd_score_norm, 3),
            "content_score": round(content_score, 3),
            "hybrid_score": round(hybrid_score, 3)
        })

    recommendations_df = pd.DataFrame(recommendations)

    if recommendations_df.empty:
        return pd.DataFrame({
            "message": [f"Không còn phim phù hợp để gợi ý cho user {user_id}."]
    })

    return (
        recommendations_df
        .sort_values(by="hybrid_score", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


# ===============================
# 7. GIAO DIỆN STREAMLIT
# ===============================
st.title("Hệ thống gợi ý phim Hybrid Recommendation")

st.write(
    """
    Hệ thống nhận đầu vào là `userId`, sau đó phân tích lịch sử rating của người dùng
    để tạo hồ sơ sở thích. Cuối cùng, hệ thống kết hợp SVD và Content-based Filtering
    để gợi ý danh sách phim phù hợp.
    """
)

st.sidebar.header("Cấu hình gợi ý")

available_user_ids = sorted(ratings_train["userId"].unique())

user_id = st.sidebar.selectbox(
    "Chọn userId",
    available_user_ids
)

top_n = st.sidebar.slider(
    "Số lượng phim gợi ý",
    min_value=5,
    max_value=30,
    value=10,
    step=5
)

alpha = st.sidebar.slider(
    "Hệ số alpha",
    min_value=0.0,
    max_value=1.0,
    value=0.7,
    step=0.1
)

min_rating_for_profile = st.sidebar.slider(
    "Ngưỡng rating để xác định phim yêu thích",
    min_value=0.5,
    max_value=5.0,
    value=3.5,
    step=0.5
)


# ===============================
# 8. HIỂN THỊ THÔNG TIN USER
# ===============================
st.subheader(f"Thông tin người dùng: User {user_id}")

user_history = ratings_train[ratings_train["userId"] == user_id]

col1, col2, col3 = st.columns(3)

col1.metric("Số phim đã rating", len(user_history))

if len(user_history) > 0:
    col2.metric("Rating trung bình", round(user_history["rating"].mean(), 2))
else:
    col2.metric("Rating trung bình", 0)

col3.metric(
    "Số phim rating cao",
    len(user_history[user_history["rating"] >= min_rating_for_profile])
)


# ===============================
# 9. HIỂN THỊ GU THỂ LOẠI
# ===============================
st.subheader("Các thể loại người dùng có xu hướng thích")

favorite_genres = get_user_favorite_genres(
    user_id=user_id,
    ratings_df=ratings_train,
    movies_df=movies_nlp,
    min_rating=min_rating_for_profile
)

if favorite_genres.empty:
    st.warning("User chưa có đủ phim rating cao để phân tích thể loại yêu thích.")
else:
    st.dataframe(favorite_genres.head(10), use_container_width=True)


# ===============================
# 10. HIỂN THỊ KẾT QUẢ GỢI Ý
# ===============================
st.subheader("Top phim gợi ý theo Hybrid Recommendation")

recommendations = recommend_movies_for_user(
    user_id=user_id,
    ratings_df=ratings_train,
    top_n=top_n,
    alpha=alpha,
    min_rating_for_profile=min_rating_for_profile
)

st.dataframe(recommendations, use_container_width=True)


# ===============================
# 11. GIẢI THÍCH CÔNG THỨC
# ===============================
st.subheader("Công thức Hybrid")

st.markdown(
    f"""
**HybridScore = alpha × SVDScore_norm + (1 - alpha) × ContentScore**

    Với cấu hình hiện tại:

    - `alpha = {alpha}`
    - Trọng số SVD = `{alpha}`
    - Trọng số nội dung = `{round(1 - alpha, 2)}`
    - `SVDScore_norm`: điểm dự đoán rating từ SVD sau khi chuẩn hóa về khoảng 0 đến 1.
    - `ContentScore`: điểm tương đồng giữa hồ sơ sở thích người dùng và nội dung phim.
    """
)