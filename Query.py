import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver import Firefox
from selenium.webdriver.firefox.options import Options
options = Options()
options.set_preference('network.trr.mode', 3)
options.set_preference('network.trr.uri', 'https://mozilla.cloudflare-dns.com/dns-query')
driver = Firefox(options=options)

file = open('Websites.txt', 'r')

while True:
    website = file.readline()
    if not website:
        break
    driver = webdriver.Firefox(executable_path=r'C:\Users\garre\OneDrive\Documents\School\DoH Research\geckodriver.exe')
    driver.get("https://www." + website.strip())
    time.sleep(3)
    print(website.strip())
    print(driver.title)
    driver.close()
    
file.close()


driver = webdriver.Chrome('./chromedriver')
driver.get("https://www.maine.com")
print(driver.title)
driver.close()