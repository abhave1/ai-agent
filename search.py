from googlesearch import search

def google_search(query):
    """
    Perform a Google search and return results
    
    Args:
        query (str): Search query
        num_results (int): Number of results to return (default: 10)
        
    Returns:
        list: List of URLs from search results
    """
    num_results = 10
    try:
        search_results = []
        for url in search(query, num_results=num_results):
            search_results.append(url)
        return search_results
    except Exception as e:
        print(f"An error occurred: {e}")
        return []