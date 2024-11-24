import json
import random
import time
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException, NoSuchElementException, InvalidArgumentException, ElementClickInterceptedException
from selenium.webdriver.chrome.options import Options
from underthesea import word_tokenize
import numpy as np
from openai import OpenAI
import os
from qdrant_client.http.models import PointStruct
from selenium_stealth import stealth
import undetected_chromedriver as uc
from loguru import logger
from DrissionPage import ChromiumPage
from DrissionPage.errors import ElementNotFoundError
from jobProcessingService import load_jobs_from_file, insert_jobs_into_qdrant
import uuid
# from fp.fp import FreeProxy
# from fake_useragent import UserAgent
# initializing a list with two User Agents
# useragentarray = [
#     "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
#     "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36",
# ]

# chrome_driver_path = r'D:\\chromedriver.exe'
chrome_driver_path = 'D:/chromedriver.exe'
# chrome_driver_path = 'D:\\chromedriver.exe'

# Create a mapping dictionary
key_mapping = {
    "job_description": "job_description",
    "requirements": "requirements",
    "benefits": "benefits",
    "location": "location",
    "working_hours": "working_hours",
    "how_to_apply": "how_to_apply",
    "end_date": "end_date",
}
# service = Service(chrome_driver_path)
# options = Options()
# options.add_argument("--disable-blink-features=AutomationControlled")

# turn-off userAutomationExtension
# options.add_experimental_option("useAutomationExtension", False)
# options.add_argument("--enable-javascript")
# options.add_argument("--enable-cookies")
# options.add_argument('--no-sandbox')
# options.add_argument('--start-maximized')
# options.add_argument('--start-fullscreen')
# options.add_argument('--disable-blink-features=AutomationControlled')
# options.add_experimental_option('useAutomationExtension', False)
# options.add_experimental_option("excludeSwitches", ["enable-automation"])
# options.add_argument("disable-infobars")
# options.add_argument("--log-level=1")

# driver = webdriver.Chrome(service=service, options=options)
driver  = ChromiumPage()



# Create a mapping dictionary
key_mapping = {
    "Mô tả công việc": "Description",
    "Yêu cầu ứng viên": "Requirements",
    "Quyền lợi": "Benefits",
    "Địa điểm làm việc": "Address",
    "Thời gian làm việc": "WorkingHours",
    "Cách thức ứng tuyển": "HowToApply",
    "Hạn nộp hồ sơ": "ApplicationDeadline",
}




# def load_url_topCv(url):
#     # Selenium
#     driver.get(url)
#     list_job_information = []

#     links_company = []

#     # just craw 10 page
#     x=0
#     while x<1:

#         try:
#             #Get the review details here
#             print('run untill here')
#             # get title
#             titleLinkContainner = driver.find_elements(by=By.XPATH, value='//div[@class="title-block"]')
#             links_company = get_links_company(titleLinkContainner)

#             #IT Recruitment Intern
#             # title = get_titles(titleLinkContainner)

#         except:
#             print('No job available')
#             break

#         # Get job information

#         #Check for button next-pagination-item have disable attribute then jump from loop else click on the next button
#         x = x+1

#     # driver.close()
#     return list_job_information


def setup_file(filename, is_append):
    if is_append:
        mod = "a+"
        bra = ']'
    else:
        mod = "w"
        bra = '['
    with open(filename, mod) as f:
        f.writelines(bra)

def write_file(filename, data, deli):
    with open(filename, "a+", encoding="utf-8") as f:
        f.writelines(deli)
        json.dump(data, f, indent=2, ensure_ascii=False)


def add_contents(contents, data):
    for header in contents.eles('xpath://div[@class="job-description__item"]//h3'):
        next_sibling_xpath = "xpath:./following-sibling::*[not(self::h3)][string-length(normalize-space()) > 0][1]"
        header_text = header.text.strip()

        try:
            next_element = header.ele(next_sibling_xpath)
            h3_header = ""
            for vn_key, en_key in key_mapping.items():
                if header_text == vn_key:
                    data[en_key] = next_element.text.strip()

            # name = data.get("name", "")
            # location = data.get("location", "")
            data['JobId'] = str(uuid.uuid4())


        except ElementNotFoundError:
            # No next sibling found or it's empty, skip this header
            print(f"No next sibling found for header '{header_text}', skipping to next header.")
            pass

def crawl_contents(filename, company_block):
    setup_file(filename, False)
    deli = ""
    print("Crawling contents...")

    for block in company_block:
        print("Crawling block...")
        link_company = ""
        time.sleep(random.uniform(3, 5))

        # Save the original window handle
        original_window = driver.get_tab()

        try:
            # Extract the link from the block
            link_element = block.ele('xpath://div[@class="avatar"]')
            try:
                print("Clicked on the link element.")
                print(link_element)

                a_tag = link_element.ele('xpath://a')
                try:
                    if a_tag:
                        link_company = a_tag.attr('href')
                        print(link_company)
                        print(f"Company link: {link_company}")
                except ElementNotFoundError:
                    print("Company link, skipping to next block.")
                    continue

                link_element.click()
            except ElementClickInterceptedException:
                print("Element click intercepted, skipping to next block.")
                continue
        except ElementNotFoundError:
            print("Link element not found in block, skipping to next block.")
            continue

        time.sleep(random.uniform(3, 5))

        # Get the list of window handles
        latest_tab = driver.latest_tab

        # Switch to the latest opened window
        driver.get_tab(latest_tab)
        print("Switched to new window.")
        time.sleep(5)

#   todo:      # Check if the page is a Cloudflare verify page

# end-todo
        try:
            print("load content")

            # Scroll down the page to simulate human behavior
            # driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            # time.sleep(2)
            tabContent = driver.latest_tab
            try:
                img = tabContent.ele('xpath://a[@class="company-logo"]')
                try:
                    if img:
                        name = img.attr("data-original-title")
                        print(name)
                except ElementNotFoundError:
                    print("Company name attribute not found, skipping to next block.")
                    driver.close_tabs(tabContent)
                    driver.get_tab(original_window)
                    continue
            except ElementNotFoundError:
                print("Company logo element not found or attribute not found, skipping to next block.")
                # driver.close_tabs(tabContent)
                # driver.get_tab(original_window)
                continue

            try:
                contents = tabContent.ele('xpath://div[@class="job-detail__information-detail--content"]')
            except ElementNotFoundError:
                print("Job detail content element not found, skipping to next block.")
                driver.close_tabs(tabContent)
                driver.get_tab(original_window)
                continue

            try:
                xpath = "xpath://div[contains(@class, 'job-detail__info--deadline')]"
                date_element = tabContent.ele(xpath)
                date_text = date_element.text
            except ElementNotFoundError:
                print("Job deadline element not found, skipping to next block.")
                driver.close_tabs(tabContent)
                driver.get_tab(original_window)
                continue

            data = {}
            data['Name'] = name
            data['ApplicationDeadline'] = date_text.split(": ")[-1].strip()
            data['LinkCompany'] = link_company
            add_contents(contents, data)
            print(data)
            write_file(filename, data, deli)
            deli = ",\n"

            time.sleep(random.uniform(3, 5))

            driver.close_tabs(tabContent)
            driver.get_tab(original_window)

            print("Closed new window.")
            deli = ",\n"
        except ElementNotFoundError:
            print("Element not found, skipping to next block.")
            driver.close_tabs(tabContent)
            driver.get_tab(original_window)
            continue
    setup_file(filename, True)
    filename = "recruit_" + "1" + "_" + "9" + ".json"
    jobs = load_jobs_from_file(filename)
    insert_jobs_into_qdrant(jobs)


# def get_links_company(homepageUrl):
#     driver.get(homepageUrl)
#     links_company = []
#     titleLinkContainner = driver.find_elements(by= xpath:value='//div[@class="title-block"]//h3[@class="title "]//a')

#     # Scroll down the page to simulate human behavior
#     driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
#     time.sleep(2)
#     for link_company in titleLinkContainner:
#         # link_obj = link_company.find_element(by=By.XPATH, value=".//a")
#         links_company.append(link_company.get_attribute("href"))

#     return links_company

def get_block_company(homepageUrl):
    driver.get(homepageUrl)
    # companyBlocks = driver.eles(By.XPATH,"//div[contains(@class, 'job-item-search-result') and contains(@class, 'bg-highlight') and contains(@class, 'job-ta')]")
    tab1 = driver.latest_tab
    companyBlocks = tab1.eles("xpath://div[contains(@class, 'job-item-search-result') and contains(@class, 'bg-highlight') and contains(@class, 'job-ta')]")
    # print(companyBlocks)

    return companyBlocks

# def get_titles(list_link):
#     titleLinkContainner = driver._find_elements(By.XPATH, '//div[@class="title-block"]')
#     titles = []
#     for title in list_link:

#         # print(title)
#         titleLocal = title._find_elements(by=By.XPATH, ".//h3[@class='title ']//a//span")
#         # print(titleLocal.text)
#         titles.append(titleLocal.text)
#         # for tit in title:
#         #     titles.append(tit)
#     return titles


if __name__ == "__main__":
    # create parser
    print("Parsing Args")

    print("Start crawling from ", 1, " to ", 2)

    links = "https://www.topcv.vn/tim-viec-lam-it-phan-mem-c10026?salary=0&exp=0&company_field=0&sort=up_top&page=1"
    company_block = get_block_company(links)
    # print(company_block)
    filename = "recruit_" + "1" + "_" + "9" + ".json"
    crawl_contents(filename, company_block)


def beginCrawlData():
    print("Start crawling from ", 1, " to ", 2)

    links = "https://www.topcv.vn/tim-viec-lam-it-phan-mem-c10026?salary=0&exp=0&company_field=0&sort=up_top&page=1"
    company_block = get_block_company(links)
    # print(company_block)
    filename = "recruit_" + "1" + "_" + "9" + ".json"
    crawl_contents(filename, company_block)
