"""
Name Entity Wordcount using PySpark + NLTK
"""

import os
import re
<<<<<<< HEAD
import gc
=======
>>>>>>> f23192f (assignment2 -> friend_recommender system added)
from pyspark.sql import SparkSession
import urllib.request
import nltk

<<<<<<< HEAD
GUTENBURG_URL = "https://www.gutenberg.org/cache/epub/100/pg100.txt"
TOP_N = 50

=======
# Config
GUTENBURG_URL = ("https://www.gutenberg.org/cache/epub/100/pg100.txt")
TOP_N = 50

# Function to download gutenberg text
>>>>>>> f23192f (assignment2 -> friend_recommender system added)
def download_gutenberg_text():
    print(f"Downloading Gutenberg text from: {GUTENBURG_URL}")
    with urllib.request.urlopen(GUTENBURG_URL) as response:
        text = response.read().decode('utf-8')
    return text

<<<<<<< HEAD
=======
# Named Entity Extraction
>>>>>>> f23192f (assignment2 -> friend_recommender system added)
def extract_entity_from_partition(lines):
    import nltk
    from nltk import sent_tokenize, word_tokenize, pos_tag, ne_chunk, Tree
    
    nltk_data = os.path.expanduser("~/nltk_data")
    if nltk_data not in nltk.data.path:
        nltk.data.path.append(nltk_data)

<<<<<<< HEAD
    # Process exactly one line at a time to prevent RAM explosion
=======
    from nltk import sent_tokenize, word_tokenize, pos_tag, ne_chunk, Tree

    # Process a smaller group of lines together to prevent OOM crashes
    buffer = []

>>>>>>> f23192f (assignment2 -> friend_recommender system added)
    for line in lines:
        line = line.strip()
        if not line:
            continue
<<<<<<< HEAD
            
        try:
            sentences = sent_tokenize(line)
=======
        
        buffer.append(line)

        # Reduced buffer from 100 to 20 to prevent Memory/Connection Reset errors
        if len(buffer) >= 20:
            text = " ".join(buffer)

            try:
                sentences = sent_tokenize(text)

                for sentence in sentences:
                    tokens = word_tokenize(sentence)

                    if not tokens:
                        continue

                    tagged_tokens = pos_tag(tokens)
                    tree = ne_chunk(tagged_tokens, binary=False)

                    for subtree in tree:
                        if isinstance(subtree, Tree):
                            entity = " ".join(
                                word for word, tag in subtree.leaves()
                            )
                            entity = entity.strip()
                            if entity:
                                yield entity
            except Exception as error:
                print("NER error:", error)

            buffer = []
    
    # Process remaining lines
    if buffer:
        text = " ".join(buffer)
        try: 
            sentences = sent_tokenize(text)
>>>>>>> f23192f (assignment2 -> friend_recommender system added)
            for sentence in sentences:
                tokens = word_tokenize(sentence)
                if not tokens:
                    continue

                tagged_tokens = pos_tag(tokens)
                tree = ne_chunk(tagged_tokens, binary=False)

                for subtree in tree:
                    if isinstance(subtree, Tree):
<<<<<<< HEAD
                        entity = " ".join(word for word, tag in subtree.leaves())
                        entity = entity.strip()
                        if entity:
                            yield entity
        except Exception:
            pass # Ignore malformed sentences silently

    # Force memory cleanup after finishing a partition
    gc.collect()

def normalize_entity(entity):
    entity = entity.strip()
    entity = re.sub(r"\s+", " ", entity)
=======
                        entity = " ".join(
                            word for word, tag in subtree.leaves()
                        )
                        entity = entity.strip()
                        if entity:
                            yield entity
        except Exception as error:
            print("NER error:", error)

# Normalize entities
def normalize_entity(entity):
    """Normalize entity text so that the same entity is counted together"""
    entity = entity.strip()

    # Remove excessive whitespace
    entity = re.sub(r"\s+", " ", entity)

    # Ignore entities that contain no alphabetic characters
>>>>>>> f23192f (assignment2 -> friend_recommender system added)
    if not re.search(r"[A-Za-z]", entity):
        return None
    return entity

<<<<<<< HEAD
def main():
    # Pre-download required NLTK resources
=======
# Main
def main():
    # Pre-download required NLTK resources silently before workers start
>>>>>>> f23192f (assignment2 -> friend_recommender system added)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    nltk.download('maxent_ne_chunker_tab', quiet=True)
    nltk.download('words', quiet=True)

<<<<<<< HEAD
    spark = (
        SparkSession.builder
        .appName("NamedEntityWordCount")
        .master("local[*]") 
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "4g")
        .config("spark.python.worker.memory", "2g")
=======
    # Create spark session with increased memory configurations
    spark = (
        SparkSession.builder
        .appName("NamedEntityWordCount")
        .master("local[*]")
        .config("spark.executor.memory", "4g")
        .config("spark.driver.memory", "4g")
        .config("spark.python.worker.memory", "1g")
        .config("spark.python.worker.faulthandler.enabled", "true")
>>>>>>> f23192f (assignment2 -> friend_recommender system added)
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")
    sc = spark.sparkContext

    print("="*50)
    print("Starting Named Entity Word Count")
    print()

    text = download_gutenberg_text()
    print("Downloaded text length:", len(text))

<<<<<<< HEAD
    # Clean header and footer
    start_pos = text.find("*** START OF THE PROJECT GUTENBERG EBOOK")
    end_pos = text.find("*** END OF THE PROJECT GUTENBERG EBOOK")
    if start_pos != -1:
        text = text[text.find("\n", start_pos) + 1:]
    if end_pos != -1:
        text = text[:end_pos]
=======
    # Remove project Gutenberg header and footer
    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"
    
    start_position = text.find(start_marker)
    end_position = text.find(end_marker)
>>>>>>> f23192f (assignment2 -> friend_recommender system added)

    # capping the number of lines to 1000 to prevent RAM explosion
    lines = text.splitlines()[:1000]
    text_rdd = sc.parallelize(lines, numSlices=4) # 4 partitions to prevent RAM explosion

<<<<<<< HEAD
    print("Spark partitions:", text_rdd.getNumPartitions())
    print("\nExtracting named entities (This will take a few minutes)...")
=======
    # Create Spark RDD
    lines = text.splitlines()
    text_rdd = sc.parallelize(
        lines,
        numSlices=8
    )

    print("Spark partitions:", text_rdd.getNumPartitions())

    # Extract named entities
    print()
    print("Extracting named entities...")
    print()
>>>>>>> f23192f (assignment2 -> friend_recommender system added)

    entities = (
        text_rdd
        .mapPartitions(extract_entity_from_partition)
        .map(normalize_entity)
        .filter(lambda entity: entity is not None)
    )

<<<<<<< HEAD
    mapped_entities = entities.map(lambda entity: (entity, 1))
    reduced_entities = mapped_entities.reduceByKey(lambda a, b: a + b)
=======
    # Map
    mapped_entities = entities.map(lambda entity: (entity, 1))

    # Let the user know processing has started (take(10) forces execution)
    print("Example output: ")
    print(mapped_entities.take(10))

    # Reduce --> combine counts for each named entity
    reduced_entities = mapped_entities.reduceByKey(lambda a, b: a + b)

    # Sort by count in descending order
>>>>>>> f23192f (assignment2 -> friend_recommender system added)
    sorted_entities = reduced_entities.sortBy(lambda x: x[1], ascending=False)

    # Display results
    results = sorted_entities.take(TOP_N)

    print("\n" + "="*50)
    print("TOP NAMED ENTITIES")
    print("="*50)
<<<<<<< HEAD
    print(f"{'Rank':<6}{'Named Entity':<40}{'Count':<10}")
    print("-" * 50)

    for rank, (entity, count) in enumerate(results, start=1):
        print(f"{rank:<6}{entity:<40}{count:<10}")
        
    print("\n" + "="*50)
    print("Total unique named entities:", reduced_entities.count())
    print("="*50)

=======
    print()

    print(
        f"{'Rank':<6}"
        f"{'Named Entity':<40}"
        f"{'Count':<10}"
    )
    print("-" * 50)

    for rank, (entity, count) in enumerate(results, start=1):
        print(
            f"{rank:<6}"
            f"{entity:<40}"
            f"{count:<10}"
        )
        
    print()
    print("="*50)
    print("Total unique named entities: ")
    print(reduced_entities.count())
    print("="*50)

    # Stop spark
>>>>>>> f23192f (assignment2 -> friend_recommender system added)
    spark.stop()

if __name__ == "__main__":
    main()