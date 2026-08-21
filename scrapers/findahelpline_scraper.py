def scrape_findahelpline():
    logger.info("=" * 55)
    logger.info("  Findahelpline Scraper Starting")
    logger.info("=" * 55)

    driver = get_driver()

    # Only load India for debugging
    url = "https://findahelpline.com/countries/in"

    driver.get(url)

    import time
    time.sleep(10)      # wait long enough for JS to render

    print(driver.title)
    print(driver.current_url)

    html = driver.page_source

    print("HTML length:", len(html))

    with open("debug.html", "w", encoding="utf-8") as f:
        f.write(html)

    print("Saved debug.html")

    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")

    if not soup:
        print("Failed to load page.")
        driver.quit()
        return

    print("Soup loaded:", soup is not None)
    print("Title:", soup.title.text if soup.title else "No title")
    print("Articles:", len(soup.find_all("article")))
    print("Divs:", len(soup.find_all("div")))
    print("H2:", len(soup.find_all("h2")))

    # Save the HTML so we can inspect it
    with open("debug.html", "w", encoding="utf-8") as f:
        f.write(soup.prettify())

    print("\nSaved page HTML to debug.html")

    driver.quit()