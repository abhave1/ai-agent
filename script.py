from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import json
import logging
import multiprocessing

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('uber_scraper.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def init_driver():
    """Initialize a headless Chrome WebDriver instance."""
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument("--disable-blink-features=AutomationControlled")  # Prevent detection
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-images")  # Speed up page load by disabling images
    return webdriver.Chrome(options)

def scrape_uber_eats(restaurant_name):
    # """Scrapes menu items from Uber Eats for a given restaurant."""
    logger.info(f"Starting scraping for restaurant: {restaurant_name}")
    driver = init_driver()
    
    try:
        driver.get(f"https://www.ubereats.com/brand/{restaurant_name}")

        # Wait for the page to load
        wait = WebDriverWait(driver, 10)
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))

        # Setting up location cookie
        cookie_value = {
            "address": {
                "address1": "Sun Devil Bookstores",
                "address2": "525 E Orange St, Tempe, AZ",
                "eaterFormattedAddress": "525 E Orange St, Tempe, AZ 85287-9989, US",
                "subtitle": "525 E Orange St, Tempe, AZ",
                "title": "Sun Devil Bookstores"
            },
            "latitude": 33.418,
            "longitude": -111.93184,
            "reference": "here:pds:place:8408lxx5-d34c385a8d6b0cbfacbe15ff491e6a9d",
            "referenceType": "here_places"
        }

        driver.add_cookie({
            'name': 'uev2.loc',
            'value': json.dumps(cookie_value),
            'domain': '.ubereats.com'
        })

        # Refresh to apply the cookie
        driver.refresh()
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "[data-testid^='store-item-']")))

        # Scrape menu
        menu = []
        items = driver.find_elements(By.CSS_SELECTOR, "[data-testid^='store-item-']")

        for item in items:
            try:
                meal_name = item.find_element(By.CSS_SELECTOR, "span[data-testid='rich-text']").text
                price = item.find_element(By.XPATH, ".//span[contains(text(), '$')]").text
                menu.append({"meal_name": meal_name, "price": price})
            except Exception:
                continue  # Skip problematic items instead of failing

        # Save results
        with open(f"{restaurant_name}.json", "w") as f:
            json.dump(menu, f, indent=4)

    except Exception as e:
        logger.error(f"Error scraping {restaurant_name}: {e}")

    finally:
        driver.quit()

    return {restaurant_name: menu}  # Return scraped data

def scrape_multiple_restaurants(restaurants):
    """Runs the scraper in parallel for multiple restaurants."""
    with multiprocessing.Pool(processes=min(len(restaurants), 6)) as pool:  # Use up to 4 parallel processes
        results = pool.map(scrape_uber_eats, restaurants)

    return results

