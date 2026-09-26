import os
import urllib.request
import tarfile
import re
import math
from pyspark.sql import SparkSession

# ==========================================
# 1. Download and Extract Dataset
# ==========================================
def download_data():
    url = "http://www.cs.cmu.edu/~ark/personas/data/MovieSummaries.tar.gz"
    tar_path = "MovieSummaries.tar.gz"
    
    if not os.path.exists(tar_path):
        print("Downloading dataset...")
        urllib.request.urlretrieve(url, tar_path)
    
    if not os.path.exists("MovieSummaries/plot_summaries.txt"):
        print("Extracting dataset...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall()

# Create a sample search terms file
def create_search_file():
    queries = [
        # 5 Single term queries
        "superman",
        "vampire",
        "matrix",
        "dinosaur",
        "wizard",
        # 5 Multiple term queries
        "funny movie with action scenes",
        "scary ghost haunting house",
        "alien invasion space battle",
        "romantic comedy wedding",
        "detective solving murder mystery"
    ]
    with open("search_queries.txt", "w") as f:
        for q in queries:
            f.write(q + "\n")

# ==========================================
# 2. Text Preprocessing & Stopwords
# ==========================================
from nltk.corpus import stopwords
# Load NLTK's English stopwords into a set for fast O(1) lookups
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    # Lowercase and remove punctuation
    text = re.sub(r'[^a-z\s]', '', text.lower())
    words = text.split()
    return [w for w in words if w not in STOPWORDS and len(w) > 2]

# ==========================================
# Main PySpark Application
# ==========================================
def main():
    download_data()
    create_search_file()
    
    spark = SparkSession.builder \
        .appName("MoviePlotSearchEngine") \
        .master("local[*]") \
        .getOrCreate()
    sc = spark.sparkContext
    sc.setLogLevel("ERROR")

    # Load Movie Metadata (Wikipedia ID -> Movie Name)
    # File format: Wikipedia ID, Freebase ID, Movie name, ...
    def parse_metadata(line):
        parts = line.split('\t')
        if len(parts) >= 3:
            return (parts[0], parts[2])
        return (None, None)
        
    movie_names = sc.textFile("MovieSummaries/movie.metadata.tsv") \
                    .map(parse_metadata) \
                    .filter(lambda x: x[0] is not None) \
                    .collectAsMap()

    # Load Plot Summaries
    # File format: Wikipedia ID \t Plot Summary
    lines = sc.textFile("MovieSummaries/plot_summaries.txt")
    
    # Map to (DocID, [words])
    docs = lines.map(lambda line: line.split('\t', 1)) \
                .filter(lambda parts: len(parts) == 2) \
                .map(lambda parts: (parts[0], clean_text(parts[1])))
                
    N = docs.count() # Total number of documents

    # ==========================================
    # 3. Calculate TF-IDF using MapReduce
    # ==========================================
    # TF: Term Frequency per document
    # Format: ((word, docID), tf_value)
    term_doc_counts = docs.flatMap(lambda x: [((word, x[0]), 1) for word in x[1]]) \
                          .reduceByKey(lambda a, b: a + b)

    # Document lengths for TF normalization (optional but good practice)
    doc_lengths = term_doc_counts.map(lambda x: (x[0][1], x[1])) \
                                 .reduceByKey(lambda a, b: a + b) \
                                 .collectAsMap()
                                 
    # Calculate Normalized TF: (word, (docID, normalized_tf))
    tf = term_doc_counts.map(lambda x: (x[0][0], (x[0][1], x[1] / doc_lengths[x[0][1]])))

    # IDF: Inverse Document Frequency
    # Document frequency: number of docs containing the word
    df = term_doc_counts.map(lambda x: (x[0][0], 1)) \
                        .reduceByKey(lambda a, b: a + b)

    idf = df.map(lambda x: (x[0], math.log10(N / x[1])))

    # TF-IDF: Join TF and IDF
    # Result format: (word, (docID, tfidf_score))
    tf_idf = tf.join(idf).map(lambda x: (x[0], (x[1][0][0], x[1][0][1] * x[1][1])))
    
    # Precompute Document Norms for Cosine Similarity
    # Norm = sqrt(sum(tfidf^2))
    doc_norms = tf_idf.map(lambda x: (x[1][0], x[1][1]**2)) \
                      .reduceByKey(lambda a, b: a + b) \
                      .mapValues(math.sqrt) \
                      .collectAsMap()

    # Cache for quick querying
    tf_idf.cache()

    # ==========================================
    # 4. Search Logic
    # ==========================================
    print("\n" + "="*50)
    print("SEARCH RESULTS")
    print("="*50)
    
    with open("search_queries.txt", "r") as f:
        queries = f.read().splitlines()
        
    for query in queries:
        query_words = clean_text(query)
        if not query_words:
            continue
            
        print(f"\nQuery: '{query}'")
        
        if len(query_words) == 1:
            # (a) Single Term Query
            term = query_words[0]
            # Filter RDD for the term, sort by tf-idf descending
            results = tf_idf.filter(lambda x: x[0] == term) \
                            .map(lambda x: (x[1][0], x[1][1])) \
                            .sortBy(lambda x: x[1], ascending=False) \
                            .take(10)
            
            for rank, (doc_id, score) in enumerate(results, 1):
                name = movie_names.get(doc_id, "Unknown Title")
                print(f"  {rank}. {name} (Score: {score:.4f})")
                
        else:
            # (b) Multiple Term Query (Cosine Similarity)
            # Create a simple TF-IDF query vector (assume tf=1/len(query), idf from corpus)
            # Fetch the IDF values for our query terms locally
            query_idf = idf.filter(lambda x: x[0] in query_words).collectAsMap()
            
            query_vector = {}
            query_norm_sq = 0
            for word in query_words:
                if word in query_idf:
                    weight = (1 / len(query_words)) * query_idf[word]
                    query_vector[word] = weight
                    query_norm_sq += weight**2
            query_norm = math.sqrt(query_norm_sq) if query_norm_sq > 0 else 1
            
            if not query_vector:
                print("  No matching terms found in corpus.")
                continue

            # Filter TF-IDF corpus for ONLY the words in the query
            # Format: (docID, query_weight * doc_weight)
            dot_products = tf_idf.filter(lambda x: x[0] in query_vector) \
                                 .map(lambda x: (x[1][0], x[1][1] * query_vector[x[0]])) \
                                 .reduceByKey(lambda a, b: a + b)
                                 
            # Calculate Cosine Similarity: (dot_product) / (doc_norm * query_norm)
            cosine_similarities = dot_products.map(lambda x: (
                x[0], 
                x[1] / (doc_norms.get(x[0], 1) * query_norm)
            ))
            
            # Sort by highest cosine similarity
            results = cosine_similarities.sortBy(lambda x: x[1], ascending=False).take(10)
            
            for rank, (doc_id, score) in enumerate(results, 1):
                name = movie_names.get(doc_id, "Unknown Title")
                print(f"  {rank}. {name} (Similarity: {score:.4f})")

    spark.stop()

if __name__ == "__main__":
    main()