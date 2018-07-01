from bs4 import BeautifulSoup

import urllib.request
import re
address = "http://vision.middlebury.edu/stereo/data/"

def getPages():
	global address
	page = set()
	html = urllib.request.urlopen(address)
	#print(html.read())
	bsoj = BeautifulSoup(html, "lxml")

	links = bsoj.table.findAll('a', {"href": re.compile("\S*")})
	
	for link in links:
		#print(link)
		page.add(address + link.attrs["href"])
	
	#print(page)
	return page


def getDownloadLinks(url):
	page = set()
	html = urllib.request.urlopen(url)
	bsoj = BeautifulSoup(html, "lxml")
	#print(bsoj)
	links = bsoj.table.findAll("a", {"href": re.compile("^.*(?!html)$")})
	
	for link in links:
		page.add(url + link.attrs["href"])
	
	print(page)
	return page
	
	
def scraper():
	links = []
	datasets = getPages()
	#print(datasets)
	for link in datasets:
		links.append(getDownloadLinks(link))
		
	print(links)
	
scraper()
#getDownloadLinks("http://vision.middlebury.edu/stereo/data/scenes2014/")