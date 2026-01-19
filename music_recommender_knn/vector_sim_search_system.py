import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


class SimilarityRecommender:
    def __init__(self, metric='cosine'):
        self.metric = metric
        self.scaler = StandardScaler()
        self.nn = None
        self.is_fitted = False
    
    def fit(self, feature_matrix):
        '''
        feature matric: np.ndarray of shape (n_items, embedding_dim)
        '''

        # Normalize feature space

        X_scaled = self.scaler.fit_transform(feature_matrix)

        #Build Similarity Index

        self.nn = NearestNeighbors(
            metric = self.metric
        )

        self.nn.fit(X_scaled)

        self.is_fitted = True

    def recommend(self, query_vectors, top_k=5):
        """
        query_vectors: np.ndarray of shape (n_queries, embedding_dim)

        Returns:
        - indices: (n_queries, top_k)
        - similarity_scores: (n_queries, top_k)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling recommend().")
        
        query_scaled = self.scaler.transform(query_vectors)

        # Retrieve neighbors

        distances, indices = self.nn.kneighbors(
            query_scaled,
            n_neighbors = top_k
        )

        # convert distance -> similarity
        # For cosine: similarity = 1 -distnace

        similarity_scores = 1- distances

        return indices, similarity_scores

np.random.seed(42)

X = np.random.rand(100, 10)
queries = np.random.rand(3, 10)

recommender = SimilarityRecommender()
recommender.fit(X)

indices, scores = recommender.recommend(queries, top_k=5)

print(indices.shape)  # (3, 5)
print(scores.shape)   # (3, 5)