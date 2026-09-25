"""
Name Entity Wordcount using PySpark + NLTK
"""

import os
import re
import subprocess
from pyspark.sql import SparkSession

#Config
GUTENBURG_URL = ("https://www.gutenberg.org/cache/epub/100/pg100.txt")

TOP_N = 50

#function to download gutenberg text
def download_gutenberg_text():
    print("Downloading Gutenberg text from:", GUTENBURG_URL)

    result = subprocess.run(
        ["wget", "-qO-", GUTENBURG_URL],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True
    )

    print("Download complete.")
    return result.stdout

#Named Entity Extraction
def extract_entity_from_partition(lines):
    import nltk
    nltk_data = os.path.expanduser("~/nltk_data")

    if nltk_data not in nltk.data.path:
        nltk.data.path.append(nltk_data)

    from nltk import sent_tokenize
    from nltk import word_tokenize
    from nltk import pos_tag
    from nltk import ne_chunk
    from nltk import Tree

    #Process a group of lines together
    buffer = []

    for line in lines:
        line = line.strip()

        if not line:
            continue
        
        buffer.append(line)

        if len(buffer) >= 100:
            text = " ".join(buffer)

            try:
                sentences = sent_tokenize(text)

                for sentence in sentences:
                    tokens = word_tokenize(sentence)

                    if not tokens:
                        continue

                    tagged_tokens = pos_tag(tokens)
                    tree = ne_chunk(
                        tagged_tokens,
                        binary=False
                    )

                    for subtree in tree:
                        if isinstance(subtree, Tree):
                            entity = " ".join(
                                word
                                for word, tag in subtree.leaves()
                            )

                            entity = entity.strip()
                            if entity:
                                yield entity
            except Exception as error:
                print(
                    "NER error:",
                    error
                )

            buffer = []
    
    #Process remaining lines
    if buffer:
        text = " ".join(buffer)
        try: 
            sentences = sent_tokenize(text)
            for sentence in sentences:
                tokens = word_tokenize(sentence)

                if not tokens:
                    continue

                tagged_tokens = pos_tag(tokens)
                tree = ne_chunk(
                    tagged_tokens,
                    binary=False
                )

                for subtree in tree:
                    if isinstance(subtree, Tree):
                        entity = " ".join(
                            word
                            for word, tag in subtree.leaves()
                        )

                        entity = entity.strip()

                        if entity:
                            yield entity
        except Exception as error:
                    print(
                        "NER error:",
                        error
                    )

#Normalize entities
def normalize_entity(entity):
    """Normalize entity text so that the same entity is counted together"""\

    entity = entity.strip()

    #Remove excessive whitespace
    entity = re.sub(r"\s+", " ", entity)

    #Ignore entities that contain no alphabetic characters
    if not re.search(r"[A-Za-z]", entity):
        return None

    return entity

#Main
def main():
    #create spark session
    spark = (
        SparkSession.builder
        .appName("NamedEntityWordCount")
        .master("local[*]")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")
    sc = spark.sparkContext

    print("="*50)
    print("Starting Named Entity Word Count")
    print()

    print("Input URL:")
    print(GUTENBURG_URL)
    print()

    text = download_gutenberg_text()

    print("Downloaded text length:", len(text))

    #Remove project Gutenberg header and footer
    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"
    
    start_position = text.find(start_marker)
    end_position = text.find(end_marker)

    if start_position != -1:
        newline_position = text.find("\n", start_position)
        if newline_position != -1:
            text = text[newline_position + 1:]
    if end_position != -1:
        text = text[:end_position]

    #Create Spark RDD
    lines = text.splitlines()
    text_rdd = sc.parallelize(
        lines,
        numSlices=8
    )

    print(
        "Spark partitions:",
        text_rdd.getNumPartitions()
    )

    #Extract named entities
    print()
    print("Extracting named entities...")
    print()

    entities = (
        text_rdd
        .mapPartitions(extract_entity_from_partition)
        .map(normalize_entity)
        .filter(lambda entity: entity is not None)
    )

    #Map
    mapped_entities = entities.map(lambda entity: (entity, 1))

    print("Example output: ")
    print(mapped_entities.take(10))

    #Reduce --> combine counts for each named entity
    reduced_entities = (
        mapped_entities.reduceByKey(lambda a, b: a + b)
    )

    #Sort by count in descending order
    sorted_entities = (
        reduced_entities
        .sortBy(lambda x: x[1],
        ascending=False)
    )

    #display results

    results = sorted_entities.take(TOP_N)

    print()
    print("="*50)
    print("TOP NAMED ENTITIES")
    print("="*50)
    print()

    print(
        f"{'Rank':<6}"
        f"{'Named Entity':<40}"
        f"{'Count':<10}"
    )
    print("-"*50)

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

    #stop spark
    spark.stop()

if __name__ == "__main__":
    main()

            