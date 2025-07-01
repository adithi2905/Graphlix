import React, { useState, useEffect } from 'react';

function App() {
  const [movies, setMovies] = useState([]);
  const [selectedMovie, setSelectedMovie] = useState('');
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    const fetchMovies = async () => {
      try {
        const response = await fetch('http://localhost:5000/movies');
        const data = await response.json();
        setMovies(data.movies);
      } catch (err) {
        console.error(err);
        setError('Failed to fetch movies.');
      }
    };
    fetchMovies();
  }, []);

  const handleFetchRecommendations = async () => {
    if (!selectedMovie) {
      setError('Please select a movie.');
      return;
    }
    setLoading(true);
    setError('');
    try {
      const response = await fetch(`http://localhost:5000/recommend_by_movie?movie_title=${encodeURIComponent(selectedMovie)}`);
      if (!response.ok) {
        throw new Error('Failed to fetch recommendations');
      }
      const data = await response.json();
      setRecommendations(data.recommendations);
    } catch (err) {
      console.error(err);
      setError('Error fetching recommendations.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-950 via-purple-900 to-purple-950 flex flex-col items-center justify-center p-6 text-white">
      <h1 className="text-4xl font-bold mb-8">🎬 Movie Recommender System</h1>

      {/* Single Select Dropdown */}
      <select
        value={selectedMovie}
        onChange={(e) => setSelectedMovie(e.target.value)}
        className="bg-purple-800 text-white rounded-lg p-3 w-72 mb-6 shadow-lg focus:outline-none focus:ring-2 focus:ring-purple-400"
      >
        <option value="">Select a movie</option>
        {movies.map((movie, idx) => (
          <option key={idx} value={movie}>
            {movie}
          </option>
        ))}
      </select>

      {/* Get Recommendations Button */}
      <button
        onClick={handleFetchRecommendations}
        className="bg-purple-400 hover:bg-purple-500 text-white font-bold py-2 px-8 rounded-lg mb-8 transition duration-300"
      >
        More Like this
      </button>

      {/* Loading / Error / Recommendations */}
      {loading && <p className="text-gray-300 mb-4">Loading recommendations...</p>}
      {error && <p className="text-red-300 mb-4">{error}</p>}

      {recommendations.length > 0 && (
        <>
          <h2 className="text-2xl font-semibold mb-4">Top Recommendations</h2>
          <ul className="list-none p-0 space-y-2">
            {recommendations.map((movie, idx) => (
              <li key={idx} className="bg-white bg-opacity-20 p-3 rounded-lg shadow-md">
                🎥 {movie}
              </li>
            ))}
          </ul>
        </>
      )}
    </div>
  );
}

export default App;
