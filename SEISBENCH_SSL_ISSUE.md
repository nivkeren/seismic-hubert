# SeismicBenchDataset SSL Certificate Issue

## Problem

The `SeismicBenchDataset` is designed to load remote SeisBench benchmark datasets (ETHZ, GEOFON, SCEDC, iquique, etc.) by downloading them from remote servers.

When attempting to load these datasets, the download fails with:

```
ssl.SSLCertificateVerificationError: 
[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: 
self-signed certificate in certificate chain
```

## Root Cause

This error is caused by **SSL certificate verification failures**, typically due to:
- Corporate firewalls intercepting HTTPS traffic
- Antivirus software with SSL inspection
- Network proxies
- System certificate issues

## Solution

### Option 1: Use Local STEAD Dataset (Recommended for Now)

The notebook `explore_stead.ipynb` and the `STEADDataset` class work perfectly with local STEAD data:

```python
from data import STEADDataset

dataset = STEADDataset(
    hdf5_path='STEAD/merge.hdf5',
    csv_path='STEAD/merge.csv',
)
```

This works because it doesn't require downloading anything - the data is stored locally.

### Option 2: Fix SSL Certificates (For Remote SeisBench Datasets)

#### macOS

Run the Python certificate installer:

```bash
/Applications/Python\ 3.11/Install\ Certificates.command
```

Then restart your Jupyter kernel.

#### Linux

Install ca-certificates:

```bash
# Ubuntu/Debian
sudo apt-get install ca-certificates

# CentOS/RedHat
sudo yum install ca-certificates
```

#### All Platforms

Check if Python can verify SSL certificates:

```python
import ssl
import certifi

# Test SSL verification
print(ssl.get_default_verify_paths())
print(certifi.where())
```

### Option 3: Configure Corporate Proxy

If you're behind a corporate firewall, configure pip to use your proxy:

```bash
pip install --proxy [user:passwd@]proxy.server:port seisbench
```

### Option 4: Temporary Workaround (Not Recommended)

```python
import os
os.environ['REQUESTS_CA_BUNDLE'] = ''

# Then load dataset
from data import SeismicBenchDataset
dataset = SeismicBenchDataset(datasets='ethz')
```

⚠️ **This disables SSL verification and is not secure. Only use for testing.**

## What Works Now

✅ STEAD dataset (local)
✅ STEADDataset class
✅ explore_stead.ipynb notebook

## What Doesn't Work

❌ SeismicBenchDataset with remote datasets (SSL error)
❌ Remote ETHZ, GEOFON, SCEDC, etc. (SSL error)

## Recommended Next Steps

1. Use `STEADDataset` and `explore_stead.ipynb` for now
2. Try the "Fix SSL Certificates" solutions above (especially macOS certificate install)
3. If issues persist, check your network/firewall settings
4. For corporate networks, contact your IT department about SSL inspection settings

## Resources

- [SeisBench Documentation](https://seisbench.readthedocs.io/)
- [Python SSL Documentation](https://docs.python.org/3/library/ssl.html)
- [Certifi Package](https://github.com/certifi/python-certifi)
