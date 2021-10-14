import sys,os,signal
import dns.message
import requests
import base64
import json
import sys
import time
import itertools
import subprocess

#List of DoH Servers to choose from 
dohServer = {
    "libredns": "https://doh.libredns.gr/dns-query",
    "libredns-ads": "https://doh.libredns.gr/ads",
    "google": "https://dns.google/dns-query",
    "cloudflare": "https://cloudflare-dns.com/dns-query",
    "quad9": "https://dns.quad9.net/dns-query",
    "cleanbrowsing": "https://doh.cleanbrowsing.org/doh/family-filter",
    "cleanbrowsing-secure": "https://doh.cleanbrowsing.org/doh/security-filter",
    "cleanbrowsing-adult": "https://doh.cleanbrowsing.org/doh/adult-filter",
    "cira": "https://private.canadianshield.cira.ca/dns-query",
    "cira-protect": "https://protected.canadianshield.cira.ca/dns-query",
    "cira-family": "https://family.canadianshield.cira.ca/dns-query",
    "securedns": "https://doh.securedns.eu/dns-query",
    "securedns-ads": "https://ads-doh.securedns.eu/dns-query",
}

# DNS record types that can be requested
recordTypes = {"A":"A", "AAAA": "AAAA", "CNAME":"CNAME", 
             "MX":"MX", "NS":"NS", "SOA":"SOA", "SPF":"SPF", "SRV":"SRV", "TXT":"TXT", "CAA":"CAA",
             "DNSKEY": "DNSKEY", "DS":"DS"}

# Use the following server
endpoint = dohServer["cloudflare"]

# Query A records
record= recordTypes["A"]

# File with list of domains to query
filename = 'dga-list.txt'

# File to save pcap 
pcapFile = filename[:-4] + '.pcap'


#Give "others" permission to read and write file to get around tshark bug
subprocess.Popen(('touch', pcapFile))
subprocess.Popen(('chmod', 'o=rw', pcapFile))


# Open the file with the list of domains to query
domainsListFile = open(filename, 'r') 

# Maximum wait time between requests. Use this to simulate advanced botnet malware
randomWaitWindow = 0

# Start tcpdump to collect the DoH packets
# -U forces tcpdump to save each packet as it arrives in the file (otherwise it uses a bufferd)
# -w writes the output to a file
tshark_process=subprocess.Popen(('sudo', 'tshark', '-w' + pcapFile))

# Wait for tcpdump to load (may or may not be needed, but adding to make sure we get all packets)
time.sleep(3)

# If you want all DoH request to go over the same TCP connection, set reuse connection to True. 
# Otherwise, set to False and each request will use its own connection
reuseConnection=True

# Create a Session object so that we can reuse the HTTPS connection for all queries 
if reuseConnection == True:
    session = requests.Session()

# Generate a DoH query for each domain in the file
for domain2query in domainsListFile:

    # Format DNS query
    message = dns.message.make_query(domain2query, record)
    dns_req = base64.urlsafe_b64encode(message.to_wire()).decode("UTF8").rstrip("=")
    
    # Send the DNS request to the server. This also wait for the reply. If there is an error, stop.
    try:
        if reuseConnection == True:
            r = session.get(endpoint, params={"dns": dns_req},
                         headers={"Content-type": "application/dns-message"})
        else:
            r = requests.get(endpoint, params={"dns": dns_req},
                         headers={"Content-type": "application/dns-message"})

        r.raise_for_status()
    except requests.RequestException as reqerror:
        sys.stderr.write("{0}\n".format(reqerror))
        sys.exit(1)

    # Stop if the reply is not a DNS response 
    if "application/dns-message" not in r.headers["Content-Type"]:
        print("The answer from: {0} is not a DNS response!".format(endpoint))
        sys.exit(1)

    # Wait for a random time if this is simulating an advanced botnet malware
    if randomWaitWindow > 0:
        time.sleep(random.randint(0, randomWaitWindow))

# Close the file 
domainsListFile.close() 

#Stop tshark
#tshark_process.kill()
os.killpg(os.getpgid(tshark_process.pid), signal.SIGTERM)

