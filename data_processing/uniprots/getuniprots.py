import urllib,urllib2

url = 'https://www.uniprot.org/uploadlists/'

params = {
'from':'ACC',
'to':'P_REFSEQ_AC',
'format':'tab',
'query':'P13368 P20806 Q9UM73 P97793 Q17192'
}

data = urllib.urlencode(params)
request = urllib2.Request(url, data)
contact = "" # Please set a contact email address here to help us debug in case of problems (see https://www.uniprot.org/help/privacy).
request.add_header('User-Agent', 'Python %s' % contact)
response = urllib2.urlopen(request)
page = response.read(200000)
