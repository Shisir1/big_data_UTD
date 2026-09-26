"""
Name Entity Wordcount using PySpark + NLTK
"""

import os
import re
import gc
from pyspark.sql import SparkSession
import urllib.request
import nltk

GUTENBURG_URL = "https://www.gutenberg.org/cache/epub/100/pg100.txt"
TOP_N = 50

def download_gutenberg_text():
    print(f"Downloading Gutenberg text from: {GUTENBURG_URL}")
    with urllib.request.urlopen(GUTENBURG_URL) as response:
        text = response.read().decode('utf-8')
    return text

def extract_entity_from_partition(lines):
    import nltk
    from nltk import sent_tokenize, word_tokenize, pos_tag, ne_chunk, Tree
    
    nltk_data = os.path.expanduser("~/nltk_data")
    if nltk_data not in nltk.data.path:
        nltk.data.path.append(nltk_data)

    # Process exactly one line at a time to prevent RAM explosion
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        try:
            sentences = sent_tokenize(line)
            for sentence in sentences:
                tokens = word_tokenize(sentence)
                if not tokens:
                    continue

                tagged_tokens = pos_tag(tokens)
                tree = ne_chunk(tagged_tokens, binary=False)

                for subtree in tree:
                    if isinstance(subtree, Tree):
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
    if not re.search(r"[A-Za-z]", entity):
        return None
    return entity

def main():
    # Pre-download required NLTK resources
    nltk.download('punkt_tab', quiet=True)
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    nltk.download('maxent_ne_chunker_tab', quiet=True)
    nltk.download('words', quiet=True)

    spark = (
        SparkSession.builder
        .appName("NamedEntityWordCount")
        .master("local[*]") 
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "4g")
        .config("spark.python.worker.memory", "2g")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")
    sc = spark.sparkContext

    print("="*50)
    print("Starting Named Entity Word Count")
    print()

    text = download_gutenberg_text()
    print("Downloaded text length:", len(text))

    # Clean header and footer
    start_pos = text.find("*** START OF THE PROJECT GUTENBERG EBOOK")
    end_pos = text.find("*** END OF THE PROJECT GUTENBERG EBOOK")
    if start_pos != -1:
        text = text[text.find("\n", start_pos) + 1:]
    if end_pos != -1:
        text = text[:end_pos]

    # CRITICAL: 200 partitions keeps the data chunks tiny
    lines = text.splitlines()[:1000]
    text_rdd = sc.parallelize(lines, numSlices=4)

    print("Spark partitions:", text_rdd.getNumPartitions())
    print("\nExtracting named entities (This will take a few minutes)...")

    entities = (
        text_rdd
        .mapPartitions(extract_entity_from_partition)
        .map(normalize_entity)
        .filter(lambda entity: entity is not None)
    )

    mapped_entities = entities.map(lambda entity: (entity, 1))
    reduced_entities = mapped_entities.reduceByKey(lambda a, b: a + b)
    sorted_entities = reduced_entities.sortBy(lambda x: x[1], ascending=False)

    results = sorted_entities.take(TOP_N)

    print("\n" + "="*50)
    print("TOP NAMED ENTITIES")
    print("="*50)
    print(f"{'Rank':<6}{'Named Entity':<40}{'Count':<10}")
    print("-" * 50)

    for rank, (entity, count) in enumerate(results, start=1):
        print(f"{rank:<6}{entity:<40}{count:<10}")
        
    print("\n" + "="*50)
    print("Total unique named entities:", reduced_entities.count())
    print("="*50)

    spark.stop()

if __name__ == "__main__":
    main()