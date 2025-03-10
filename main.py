from script import scrape_multiple_restaurants
# from script import 
import json
import threading

# Example usage:
if __name__ == "__main__":
    restaurant_list = ["mcdonalds", "burger-king", "wendys", "taco-bell", "subway"]
    
    # Run multiprocessing
    menu_data = scrape_multiple_restaurants(restaurant_list)
    print("done")

    # # Print the results
    # for restaurant, menu in menu_data:
    #     print(f"\nMenu for {restaurant}:")
    #     print(json.dumps(menu, indent=4))