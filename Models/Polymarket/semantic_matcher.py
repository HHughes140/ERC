"""
Semantic Matcher for Cross-Platform Arbitrage

Uses sentence embeddings to match similar markets across platforms.

Key Features:
1. Semantic similarity using sentence embeddings
2. Outcome alignment detection (YES/NO inversions)
3. Confidence scoring based on match quality
4. Caching for performance

CRITICAL: Naive word overlap (30%) causes false matches!
"Trump wins 2024" and "Biden wins 2024" share many words but are OPPOSITE.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import re
import logging

logger = logging.getLogger(__name__)


@dataclass
class MarketDescription:
    """Market with text description"""
    platform: str
    market_id: str
    title: str
    description: str
    yes_price: float
    no_price: float
    volume: float


@dataclass
class MatchResult:
    """Result of semantic matching"""
    market_a: MarketDescription
    market_b: MarketDescription
    similarity_score: float       # 0-1, higher = more similar
    outcome_alignment: str        # 'same', 'opposite', 'uncertain'
    alignment_confidence: float   # Confidence in alignment determination
    is_valid_match: bool
    arbitrage_potential: float    # Potential profit if valid
    reason: str


class SimpleEmbedder:
    """
    Simple word-based embedder for when sentence-transformers isn't available.

    Uses TF-IDF weighted word vectors with basic word embeddings.
    Not as good as sentence-transformers but works without dependencies.
    """

    def __init__(self, embedding_dim: int = 100):
        self.embedding_dim = embedding_dim
        self.vocab: Dict[str, np.ndarray] = {}
        self.idf: Dict[str, float] = {}

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        tokens = text.split()
        # Remove very short tokens and stopwords
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'will', 'be',
                    'to', 'of', 'and', 'or', 'in', 'on', 'at', 'by', 'for', 'with'}
        return [t for t in tokens if len(t) > 2 and t not in stopwords]

    def _get_word_embedding(self, word: str) -> np.ndarray:
        """Get or create embedding for word"""
        if word not in self.vocab:
            # Create deterministic pseudo-random embedding from word
            np.random.seed(hash(word) % (2**32))
            self.vocab[word] = np.random.randn(self.embedding_dim)
            self.vocab[word] /= np.linalg.norm(self.vocab[word])
        return self.vocab[word]

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts to embeddings.

        Uses mean of word embeddings (simple but effective).
        """
        embeddings = []

        for text in texts:
            tokens = self._tokenize(text)

            if not tokens:
                embeddings.append(np.zeros(self.embedding_dim))
                continue

            # Average word embeddings
            word_vecs = [self._get_word_embedding(t) for t in tokens]
            embedding = np.mean(word_vecs, axis=0)

            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            embeddings.append(embedding)

        return np.array(embeddings)


class SemanticMatcher:
    """
    Matches markets across platforms using semantic similarity.

    Strategy:
    1. Encode market titles/descriptions to embeddings
    2. Calculate cosine similarity
    3. Check for outcome alignment (same YES or opposite)
    4. Validate match quality
    """

    def __init__(self, similarity_threshold: float = 0.80,
                 use_transformers: bool = True):
        """
        Args:
            similarity_threshold: Minimum similarity to consider a match
            use_transformers: Try to use sentence-transformers (better quality)
        """
        self.similarity_threshold = similarity_threshold
        self.embedder = None
        self._embedding_cache: Dict[str, np.ndarray] = {}

        # Try to load sentence-transformers
        if use_transformers:
            try:
                from sentence_transformers import SentenceTransformer
                self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Using sentence-transformers for semantic matching")
            except ImportError:
                logger.warning("sentence-transformers not available, using simple embedder")
                self.embedder = SimpleEmbedder()
        else:
            self.embedder = SimpleEmbedder()

        # Keywords that indicate opposite outcomes
        self.opposite_indicators = {
            'win': 'lose',
            'wins': 'loses',
            'victory': 'defeat',
            'yes': 'no',
            'pass': 'fail',
            'above': 'below',
            'over': 'under',
            'more': 'less',
            'higher': 'lower',
            'up': 'down',
            'rise': 'fall',
            'increase': 'decrease',
            'positive': 'negative',
            'gain': 'loss',
        }

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding with caching"""
        if text not in self._embedding_cache:
            if hasattr(self.embedder, 'encode'):
                if isinstance(self.embedder, SimpleEmbedder):
                    embedding = self.embedder.encode([text])[0]
                else:
                    embedding = self.embedder.encode([text], convert_to_numpy=True)[0]
            else:
                embedding = self.embedder.encode([text])[0]
            self._embedding_cache[text] = embedding

        return self._embedding_cache[text]

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between vectors"""
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (norm_a * norm_b)

    def _check_outcome_alignment(self, text_a: str, text_b: str) -> Tuple[str, float]:
        """
        Determine if two markets have same or opposite outcomes.

        Returns: (alignment, confidence)
        - 'same': YES in A = YES in B
        - 'opposite': YES in A = NO in B
        - 'uncertain': Can't determine
        """
        text_a_lower = text_a.lower()
        text_b_lower = text_b.lower()

        opposite_score = 0
        same_score = 0

        # Check for opposite keywords
        for word_a, word_b in self.opposite_indicators.items():
            if word_a in text_a_lower:
                if word_b in text_b_lower:
                    opposite_score += 1
                elif word_a in text_b_lower:
                    same_score += 1

        # Check for negation patterns
        negation_patterns = [
            (r'\bnot\b', r'\bnot\b'),  # Both have 'not' = same
            (r'\bwon\'t\b', r'\bwill\b'),  # Opposite
            (r'\bno\b', r'\byes\b'),  # Opposite
        ]

        for pat_a, pat_b in negation_patterns:
            if re.search(pat_a, text_a_lower) and re.search(pat_b, text_b_lower):
                opposite_score += 0.5

        # Determine alignment
        total = same_score + opposite_score

        if total == 0:
            return 'uncertain', 0.5

        if same_score > opposite_score:
            confidence = same_score / total
            return 'same', min(confidence, 0.95)
        else:
            confidence = opposite_score / total
            return 'opposite', min(confidence, 0.95)

    def _extract_key_entities(self, text: str) -> List[str]:
        """Extract key entities (names, numbers, dates) from text"""
        entities = []

        # Numbers (years, percentages, etc.)
        numbers = re.findall(r'\b\d+(?:\.\d+)?%?\b', text)
        entities.extend(numbers)

        # Capitalized words (likely names)
        caps = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        entities.extend(caps)

        return entities

    def match_markets(self, market_a: MarketDescription,
                      market_b: MarketDescription) -> MatchResult:
        """
        Determine if two markets are semantically the same.

        Args:
            market_a: First market
            market_b: Second market

        Returns:
            MatchResult with similarity and alignment info
        """
        # Combine title and description for richer comparison
        text_a = f"{market_a.title}. {market_a.description}"
        text_b = f"{market_b.title}. {market_b.description}"

        # Get embeddings
        emb_a = self._get_embedding(text_a)
        emb_b = self._get_embedding(text_b)

        # Calculate similarity
        similarity = self._cosine_similarity(emb_a, emb_b)

        # Check key entity overlap (important sanity check)
        entities_a = set(self._extract_key_entities(text_a))
        entities_b = set(self._extract_key_entities(text_b))

        if entities_a and entities_b:
            entity_overlap = len(entities_a & entities_b) / max(len(entities_a), len(entities_b))
        else:
            entity_overlap = 0.5  # Neutral if no entities found

        # Penalize low entity overlap
        if entity_overlap < 0.3:
            similarity *= 0.7

        # Check outcome alignment
        alignment, alignment_conf = self._check_outcome_alignment(
            market_a.title, market_b.title
        )

        # Determine validity
        is_valid = similarity >= self.similarity_threshold and entity_overlap >= 0.3

        # Calculate arbitrage potential
        if is_valid:
            if alignment == 'same':
                # Same outcome: buy cheaper YES
                arb_potential = abs(market_a.yes_price - market_b.yes_price)
            elif alignment == 'opposite':
                # Opposite: A's YES = B's NO
                combined_cost = market_a.yes_price + market_b.no_price
                arb_potential = max(0, 1 - combined_cost)
            else:
                arb_potential = 0
        else:
            arb_potential = 0

        # Generate reason
        if is_valid:
            reason = f"Valid match: {similarity:.0%} similarity, {alignment} outcomes"
        elif similarity < self.similarity_threshold:
            reason = f"Low similarity: {similarity:.0%} < {self.similarity_threshold:.0%}"
        elif entity_overlap < 0.3:
            reason = f"Key entity mismatch: {entity_overlap:.0%} overlap"
        else:
            reason = "Unknown reason"

        return MatchResult(
            market_a=market_a,
            market_b=market_b,
            similarity_score=similarity,
            outcome_alignment=alignment,
            alignment_confidence=alignment_conf,
            is_valid_match=is_valid,
            arbitrage_potential=arb_potential,
            reason=reason
        )

    def find_matches(self, markets_a: List[MarketDescription],
                     markets_b: List[MarketDescription],
                     top_k: int = 10) -> List[MatchResult]:
        """
        Find best matches between two lists of markets.

        Args:
            markets_a: Markets from platform A
            markets_b: Markets from platform B
            top_k: Number of top matches to return

        Returns:
            List of MatchResults sorted by arbitrage potential
        """
        all_matches = []

        for market_a in markets_a:
            for market_b in markets_b:
                # Skip same platform
                if market_a.platform == market_b.platform:
                    continue

                result = self.match_markets(market_a, market_b)

                if result.is_valid_match and result.arbitrage_potential > 0.01:
                    all_matches.append(result)

        # Sort by arbitrage potential
        all_matches.sort(key=lambda x: x.arbitrage_potential, reverse=True)

        return all_matches[:top_k]

    def build_similarity_matrix(self, markets: List[MarketDescription]) -> np.ndarray:
        """
        Build pairwise similarity matrix for all markets.

        Useful for clustering or visualization.
        """
        n = len(markets)
        texts = [f"{m.title}. {m.description}" for m in markets]

        # Get all embeddings
        embeddings = np.array([self._get_embedding(t) for t in texts])

        # Calculate similarity matrix
        similarity = np.zeros((n, n))

        for i in range(n):
            for j in range(i, n):
                sim = self._cosine_similarity(embeddings[i], embeddings[j])
                similarity[i, j] = sim
                similarity[j, i] = sim

        return similarity


def test_semantic_matcher():
    """Test the semantic matcher"""
    print("=== Semantic Matcher Test ===\n")

    matcher = SemanticMatcher(similarity_threshold=0.75, use_transformers=False)

    # Test markets
    markets = [
        MarketDescription(
            platform="polymarket",
            market_id="poly_1",
            title="Will Trump win the 2024 presidential election?",
            description="Market resolves YES if Donald Trump wins the 2024 US presidential election.",
            yes_price=0.45,
            no_price=0.55,
            volume=1000000
        ),
        MarketDescription(
            platform="kalshi",
            market_id="kalshi_1",
            title="Trump wins 2024 election",
            description="Will Donald Trump be elected president in 2024?",
            yes_price=0.48,
            no_price=0.52,
            volume=500000
        ),
        MarketDescription(
            platform="polymarket",
            market_id="poly_2",
            title="Will Biden win the 2024 presidential election?",
            description="Market resolves YES if Joe Biden wins the 2024 US presidential election.",
            yes_price=0.35,
            no_price=0.65,
            volume=800000
        ),
        MarketDescription(
            platform="kalshi",
            market_id="kalshi_2",
            title="Bitcoin above $100K by end of 2024",
            description="Will Bitcoin reach $100,000 before January 1, 2025?",
            yes_price=0.20,
            no_price=0.80,
            volume=200000
        ),
    ]

    # Test pairwise matching
    print("Pairwise Matching Results:")
    print("-" * 60)

    for i, market_a in enumerate(markets):
        for market_b in markets[i+1:]:
            result = matcher.match_markets(market_a, market_b)
            print(f"\n{market_a.title[:40]}...")
            print(f"vs {market_b.title[:40]}...")
            print(f"  Similarity: {result.similarity_score:.1%}")
            print(f"  Alignment: {result.outcome_alignment} ({result.alignment_confidence:.0%})")
            print(f"  Valid Match: {result.is_valid_match}")
            if result.arbitrage_potential > 0:
                print(f"  Arb Potential: ${result.arbitrage_potential:.2f}")

    # Test bulk matching
    print("\n" + "=" * 60)
    print("Finding Cross-Platform Matches:")
    print("-" * 60)

    poly_markets = [m for m in markets if m.platform == "polymarket"]
    kalshi_markets = [m for m in markets if m.platform == "kalshi"]

    matches = matcher.find_matches(poly_markets, kalshi_markets, top_k=5)

    for match in matches:
        print(f"\nMatch Found:")
        print(f"  {match.market_a.platform}: {match.market_a.title[:40]}...")
        print(f"  {match.market_b.platform}: {match.market_b.title[:40]}...")
        print(f"  Similarity: {match.similarity_score:.1%}")
        print(f"  Arb Potential: ${match.arbitrage_potential:.2f}")


if __name__ == "__main__":
    test_semantic_matcher()
