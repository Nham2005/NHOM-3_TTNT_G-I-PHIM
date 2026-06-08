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


# =========================================================
# 1. LOAD MODEL VÀ DỮ LIỆU ĐÃ LƯU TỪ NOTEBOOK
# =========================================================
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

    with open("model/best_alpha.pkl", "rb") as f:
        best_alpha = pickle.load(f)

    return (
        svd_model,
        tfidf_matrix,
        movies_nlp,
        ratings_train,
        movie_id_to_index,
        best_alpha
    )


(
    svd_model,
    tfidf_matrix,
    movies_nlp,
    ratings_train,
    movie_id_to_index,
    best_alpha
) = load_artifacts()


# =========================================================
# 2. HÀM THỐNG KÊ THỂ LOẠI USER THÍCH
# =========================================================
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
        genres = str(row.genres).split()

        for genre in genres:
            if genre not in genre_stats:
                genre_stats[genre] = {
                    "ratings": [],
                    "count": 0
                }

            genre_stats[genre]["ratings"].append(row.rating)
            genre_stats[genre]["count"] += 1

    result = []

    for genre, stats in genre_stats.items():
        result.append({
            "genre": genre,
            "count": stats["count"],
            "avg_rating": round(np.mean(stats["ratings"]), 3)
        })

    return (
        pd.DataFrame(result)
        .sort_values(by=["avg_rating", "count"], ascending=False)
        .reset_index(drop=True)
    )


# =========================================================
# 3. HÀM TẠO HỒ SƠ SỞ THÍCH USER TỪ TF-IDF
# =========================================================
def build_user_profile(user_id, ratings_df, threshold=3.5):
    liked_ratings = ratings_df[
        (ratings_df["userId"] == user_id) &
        (ratings_df["rating"] >= threshold)
    ]

    if liked_ratings.empty:
        return None

    movie_vectors = []
    weights = []

    for row in liked_ratings.itertuples():
        movie_id = row.movieId

        if movie_id in movie_id_to_index:
            movie_index = movie_id_to_index[movie_id]
            movie_vectors.append(tfidf_matrix[movie_index])
            weights.append(row.rating)

    if len(movie_vectors) == 0:
        return None

    movie_matrix = vstack(movie_vectors)
    weights = np.array(weights)

    user_profile = (
        movie_matrix.multiply(weights[:, None]).sum(axis=0)
        / weights.sum()
    )

    return np.asarray(user_profile)


# =========================================================
# 4. HÀM CHUẨN HÓA ĐIỂM SVD
# =========================================================
def normalize_svd_score(score, min_rating=0.5, max_rating=5.0):
    normalized_score = (score - min_rating) / (max_rating - min_rating)
    return max(0, min(1, normalized_score))


# =========================================================
# 5. HÀM GỢI Ý PHIM HYBRID
# =========================================================
def recommend_movies_for_user(
    user_id,
    top_n=10,
    alpha=0.7,
    threshold=3.5
):
    rated_items = set(
        ratings_train[ratings_train["userId"] == user_id]["movieId"]
    )

    candidate_movies = movies_nlp[
        ~movies_nlp["movieId"].isin(rated_items)
    ].copy()

    user_profile = build_user_profile(
        user_id=user_id,
        ratings_df=ratings_train,
        threshold=threshold
    )

    if user_profile is not None:
        content_scores = cosine_similarity(
            user_profile,
            tfidf_matrix
        ).flatten()
    else:
        content_scores = None

    favorite_genres = get_user_favorite_genres(
        user_id=user_id,
        ratings_df=ratings_train,
        movies_df=movies_nlp,
        min_rating=threshold
    )

    top_genres = favorite_genres["genre"].head(3).tolist()

    recommendations = []

    for row in candidate_movies.itertuples():
        movie_id = row.movieId

        svd_pred_rating = svd_model.predict(
            user_id,
            movie_id
        ).est

        svd_score_norm = normalize_svd_score(svd_pred_rating)

        if content_scores is not None and movie_id in movie_id_to_index:
            content_score = content_scores[movie_id_to_index[movie_id]]
        else:
            content_score = 0

        hybrid_score = (
            alpha * svd_score_norm
            + (1 - alpha) * content_score
        )

        reason = "Điểm SVD cao và nội dung phù hợp với sở thích người dùng"

        if len(top_genres) > 0:
            reason = "Phù hợp với gu thể loại: " + ", ".join(top_genres)

        recommendations.append({
            "movieId": movie_id,
            "Tên phim": row.title,
            "Thể loại": row.genres,
            "Điểm SVD dự đoán": round(svd_pred_rating, 3),
            "Điểm nội dung": round(content_score, 3),
            "Điểm gợi ý": round(hybrid_score, 3),
            "Lý do gợi ý": reason
        })

    recommendations_df = pd.DataFrame(recommendations)

    if recommendations_df.empty:
        return pd.DataFrame({
            "message": [f"Không còn phim phù hợp để gợi ý cho user {user_id}."]
        })

    return (
        recommendations_df
        .sort_values("Điểm gợi ý", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


# =========================================================
# 6. GIAO DIỆN STREAMLIT
# =========================================================
st.title("Hệ thống gợi ý phim Hybrid Recommendation")


# Sidebar
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
    value=float(best_alpha),
    step=0.1
)

threshold = st.sidebar.slider(
    "Ngưỡng rating để xác định phim yêu thích",
    min_value=0.5,
    max_value=5.0,
    value=3.5,
    step=0.5
)


# Thông tin user
st.subheader(f"Thông tin người dùng: User {user_id}")

user_history = ratings_train[
    ratings_train["userId"] == user_id
]

high_rating_history = user_history[
    user_history["rating"] >= threshold
]

col1, col2, col3 = st.columns(3)

col1.metric("Số phim đã rating", len(user_history))

if len(user_history) > 0:
    col2.metric("Rating trung bình", round(user_history["rating"].mean(), 2))
else:
    col2.metric("Rating trung bình", 0)

col3.metric("Số phim rating cao", len(high_rating_history))


# Phim user đã đánh giá cao
st.subheader("Một số phim người dùng đã đánh giá cao")

user_high_movies = high_rating_history.merge(
    movies_nlp[["movieId", "title", "genres"]],
    on="movieId",
    how="left"
)

if user_high_movies.empty:
    st.warning("User chưa có phim rating cao.")
else:
    st.dataframe(
        user_high_movies[["title", "genres", "rating"]]
        .sort_values("rating", ascending=False)
        .head(10)
        .rename(columns={
            "title": "Tên phim",
            "genres": "Thể loại",
            "rating": "Rating"
        }),
        use_container_width=True
    )


# Thể loại yêu thích
st.subheader("Các thể loại người dùng có xu hướng thích")

favorite_genres = get_user_favorite_genres(
    user_id=user_id,
    ratings_df=ratings_train,
    movies_df=movies_nlp,
    min_rating=threshold
)

if favorite_genres.empty:
    st.warning("User chưa có đủ dữ liệu để phân tích thể loại yêu thích.")
else:
    favorite_genres_display = (
    favorite_genres
    .head(10)
    .rename(columns={
        "genre": "Thể loại",
        "count": "Số phim rating cao",
        "avg_rating": "Rating trung bình"
    })
)

st.dataframe(
    favorite_genres_display,
    use_container_width=True
)

# Gợi ý phim
st.subheader("Top phim gợi ý theo Hybrid Recommendation")

recommendations = recommend_movies_for_user(
    user_id=user_id,
    top_n=top_n,
    alpha=alpha,
    threshold=threshold
)

st.dataframe(
    recommendations,
    use_container_width=True
)


# Công thức
st.subheader("Công thức Hybrid")

st.markdown(
    f"""
    **Công thức tính điểm gợi ý:**

    **Điểm gợi ý (`hybrid_score`) = alpha × Điểm SVD chuẩn hóa (`svd_score_norm`) + (1 - alpha) × Điểm nội dung (`content_score`)**

    Với cấu hình hiện tại:

    - `alpha = {alpha}`
    - Trọng số SVD = `{alpha}`
    - Trọng số nội dung = `{round(1 - alpha, 2)}`
    - Ngưỡng rating yêu thích = `{threshold}`

    **Ý nghĩa các cột trong bảng gợi ý:**

    - **Tên phim** (`title`): tên bộ phim được hệ thống đề xuất cho người dùng.
    - **Thể loại** (`genres`): các thể loại của bộ phim.
    - **Điểm SVD dự đoán** (`svd_pred_rating`): điểm rating mà mô hình SVD dự đoán người dùng có thể chấm cho bộ phim, theo thang điểm từ 0.5 đến 5.0.
    - **Điểm nội dung** (`content_score`): mức độ tương đồng giữa nội dung bộ phim và hồ sơ sở thích của người dùng.
    - **Điểm gợi ý** (`hybrid_score`): điểm cuối cùng của mô hình Hybrid. Điểm này được tính bằng cách kết hợp điểm SVD chuẩn hóa (`svd_score_norm`) và điểm nội dung (`content_score`). Phim có điểm gợi ý càng cao thì càng được ưu tiên đề xuất.
    - **Lý do gợi ý** (`reason`): giải thích ngắn gọn vì sao phim được đề xuất, dựa trên thể loại hoặc sở thích nổi bật của người dùng.
    """
)
