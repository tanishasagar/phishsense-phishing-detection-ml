"""
Feature extraction for PhishSense
Run: python src/feature_extractor.py --sample
"""
import argparse
import re
import urllib.parse
from datetime import datetime
import socket, ssl, requests
from bs4 import BeautifulSoup
import tldextract, pandas as pd

def lexical_features(url):
    parsed = urllib.parse.urlparse(url if url.startswith('http') else 'http://' + url)
    f = {}
    f['url'] = url
    f['url_length'] = len(url)
    f['hostname_length'] = len(parsed.netloc)
    f['path_length'] = len(parsed.path)
    f['count_dots'] = url.count('.')
    f['count_hyphen'] = url.count('-')
    f['count_at'] = url.count('@')
    f['count_question'] = url.count('?')
    f['contains_ip'] = 1 if re.match(r'^\d+\.\d+\.\d+\.\d+', parsed.netloc) else 0
    ext = tldextract.extract(url)
    f['subdomain_count'] = 0 if ext.subdomain == '' else ext.subdomain.count('.') + 1
    f['domain'] = ext.domain + ('.' + ext.suffix if ext.suffix else '')
    return f

def whois_age_days(domain):  # placeholder
    return -1

def ssl_validity_days(hostname):
    try:
        host = hostname.split(':')[0]
        ctx = ssl.create_default_context()
        with socket.create_connection((host, 443), timeout=5) as sock:
            with ctx.wrap_socket(sock, server_hostname=host) as ssock:
                cert = ssock.getpeercert()
                not_after = cert.get('notAfter')
                if not_after:
                    exp = datetime.strptime(not_after, '%b %d %H:%M:%S %Y %Z')
                    return (exp - datetime.utcnow()).days
        return -1
    except Exception:
        return -1

def html_content_features(url):
    feats = {}
    try:
        r = requests.get(url if url.startswith('http') else 'http://' + url,
                         timeout=5, headers={'User-Agent': 'Mozilla/5.0'})
        soup = BeautifulSoup(r.text, 'lxml')
        feats['num_forms'] = len(soup.find_all('form'))
        feats['num_password_inputs'] = len(soup.find_all('input', {'type':'password'}))
        feats['num_iframes'] = len(soup.find_all('iframe'))
        text = soup.get_text(" ").lower()
        feats['contains_login_keyword'] = int(any(k in text for k in ['login','signin','password']))
    except Exception:
        feats['num_forms'] = feats['num_password_inputs'] = feats['num_iframes'] = feats['contains_login_keyword'] = -1
    return feats

def extract_features(url):
    f = lexical_features(url)
    f['whois_age_days'] = whois_age_days(f.get('domain',''))
    host = urllib.parse.urlparse(url if url.startswith('http') else 'http://' + url).netloc
    f['ssl_valid_days'] = ssl_validity_days(host)
    f.update(html_content_features(url))
    return f

def sample_run():
    urls = [
        "http://example.com",
        "https://www.paypal.com/signin",
        "http://192.168.0.1/admin",
        "http://bit.ly/2Fake"
    ]
    rows = [extract_features(u) for u in urls]
    df = pd.DataFrame(rows)
    df.to_csv("data/processed/sample_features.csv", index=False)
    print("✅ Created: data/processed/sample_features.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', action='store_true')
    args = parser.parse_args()
    if args.sample:
        sample_run()
    else:
        print("Use  --sample  to create sample features.")
