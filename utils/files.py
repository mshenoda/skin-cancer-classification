import shutil
import os
import difPy

def find_duplicates(directory: str, workers:int =8) -> difPy.search:
    """
    Finds duplicate or similar images in a directory   

    Args:
    - directory : str, list
        Paths of the directories or the files to be searched
    - workers (optional):
        Number of worker processes for multiprocessing

    Returns:
    - search: a difPy search object for matches
    """
    dif = difPy.build(directory)
    search = difPy.search(dif, similarity="similar", processes=workers)
    return search

def delete_duplicates(duplicates_dict, threshold):
    """
    Deletes duplicate files based on the provided duplicates dictionary and similarity threshold.

    Args:
    - duplicates_dict (dict): A dictionary where keys are original file paths and values are lists of tuples,
      each containing a duplicate file path and its similarity score.
    - threshold (float): The similarity score threshold above which duplicates are considered for deletion.

    Returns:
    - None: This function performs file deletions and prints the results.
    """
    high_score_duplicates = filter_duplicates(duplicates_dict, threshold)
    for original, duplicates in high_score_duplicates.items():
        for duplicate, _ in duplicates:
            try:
                # Remove the duplicate file
                if os.path.exists(duplicate):
                    os.remove(duplicate)
                    print(f"Deleted duplicate file: {duplicate}")
                else:
                    print(f"File not found: {duplicate}")
            except Exception as e:
                print(f"Error deleting file {duplicate}: {e}")

def filter_duplicates(duplicates_dict, threshold=1.0):
    """
    Filters duplicates based on the similarity score threshold.

    Args:
    - duplicates_dict (dict): A dictionary where keys are original file paths and values are lists of tuples,
      each containing a duplicate file path and its similarity score.
    - threshold (float, optional): The minimum similarity score required to consider a duplicate. Default is 1.0.

    Returns:
    - dict: A dictionary where keys are original file paths and values are lists of tuples containing
      duplicate file paths and their similarity scores, filtered by the threshold.
    """
    high_scored_duplicates = {}
    
    for original, duplicates in duplicates_dict.items():
        # Filter duplicates based on the similarity threshold
        filtered_duplicates = [
            (duplicate, score) for duplicate, score in duplicates if score >= threshold
        ]
        
        if filtered_duplicates:
            high_scored_duplicates[original] = filtered_duplicates
    
    return high_scored_duplicates