from flask import Flask, jsonify, request
from flask_cors import CORS
import torch
import pickle
import pandas as pd
import difflib
from model import EnhancedNGCF

app = Flask(__name__)
CORS(app)

# === Load graph adjacency matrix ===
try:
    adj = torch.load("graph_adj.pt", map_location=torch.device('cpu'))
except FileNotFoundError:
    raise FileNotFoundError("Missing graph_adj.pt. Please run training and save the graph.")

# === Load mappings ===
with open("user2idx.pkl", "rb") as f:
    user2idx = pickle.load(f)
with open("item2idx.pkl", "rb") as f:
    item2idx = pickle.load(f)
with open("reverse_item_map.pkl", "rb") as f:
    reverse_item_map = pickle.load(f)

num_users = len(user2idx)
num_items = len(item2idx)

# === Load model ===
model = EnhancedNGCF(num_users, num_items)
model.load_state_dict(torch.load("enhanced_ngcf_model.pth", map_location=torch.device('cpu')))
model.eval()

# === Load movie metadata ===
def load_movies(filepath='movies.dat'):
    try:
        movies = pd.read_csv(filepath, sep='::', engine='python', names=['movie_id', 'title', 'genres'], encoding='ISO-8859-1')
        title_to_id = {row['title']: row['movie_id'] for _, row in movies.iterrows()}
        id_to_title = {row['movie_id']: row['title'] for _, row in movies.iterrows()}
        return list(title_to_id.keys()), title_to_id, id_to_title
    except Exception as e:
        raise RuntimeError(f"Failed to load movies: {e}")

MOVIES, MOVIE_IDS, ID_TO_TITLE = load_movies()

# === API Routes ===
@app.route('/')
def home():
    return 'Enhanced NGCF Recommender API is running.'

@app.route('/movies', methods=['GET'])
def get_movies():
    return jsonify({'movies': MOVIES})

@app.route('/recommend_by_user', methods=['GET'])
def recommend_by_user():
    user_id_str = request.args.get('user_id')
    if not user_id_str:
        return jsonify({'error': 'Missing user_id parameter'}), 400

    try:
        user_id = int(user_id_str)
    except:
        return jsonify({'error': 'Invalid user_id format'}), 400

    if user_id not in user2idx:
        return jsonify({'error': f'User ID {user_id} not found'}), 404

    user_idx = user2idx[user_id]

    with torch.no_grad():
        user_embs, item_embs = model(adj)
        user_vector = user_embs[user_idx]
        scores = torch.matmul(user_vector, item_embs.T)
        topk_indices = torch.topk(scores, k=10).indices.tolist()
        recommended_movie_ids = [reverse_item_map[idx] for idx in topk_indices]
        recommended_titles = [ID_TO_TITLE[mid] for mid in recommended_movie_ids if mid in ID_TO_TITLE]

    return jsonify({
        'user_id': user_id,
        'recommendations': recommended_titles
    })

@app.route('/recommend_by_movie', methods=['GET'])
def recommend_by_movie():
    movie_title = request.args.get('movie_title')
    if not movie_title:
        return jsonify({'error': 'Missing movie_title parameter'}), 400

    # Fuzzy match to handle typos or formatting
    close_matches = difflib.get_close_matches(movie_title, MOVIE_IDS.keys(), n=1, cutoff=0.6)
    if not close_matches:
        return jsonify({'error': f"Movie '{movie_title}' not found"}), 404

    matched_title = close_matches[0]
    movie_id = MOVIE_IDS[matched_title]
    item_idx = item2idx.get(movie_id)

    if item_idx is None:
        return jsonify({'error': 'No index found for the matched movie'}), 404

    with torch.no_grad():
        user_embs, item_embs = model(adj)
        item_vector = item_embs[item_idx]
        scores = torch.matmul(item_vector, user_embs.T)
        topk_user_indices = torch.topk(scores, k=1).indices.tolist()
        user_idx = topk_user_indices[0]

        user_vector = user_embs[user_idx]
        scores = torch.matmul(user_vector, item_embs.T)
        topk_item_indices = torch.topk(scores, k=10).indices.tolist()

        recommended_movie_ids = [reverse_item_map[idx] for idx in topk_item_indices]
        recommended_titles = [ID_TO_TITLE[mid] for mid in recommended_movie_ids if mid in ID_TO_TITLE]

    return jsonify({
    'input_movie': str(matched_title),
    'user_like': int(list(user2idx.keys())[user_idx]),
    'recommendations': [str(title) for title in recommended_titles]
})


# === Run App ===
if __name__ == '__main__':
    app.run(debug=True)
