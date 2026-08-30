FROM dockerreg01.accounts.ad.wustl.edu/swamidass/xenosite-legacy:api

# Test-only HTTP API (not production Flask). Loads libridass already in the image.
COPY server.py /opt/legacy-test-api/server.py
ENV PORT=8099
EXPOSE 8099
CMD ["python", "/opt/legacy-test-api/server.py"]
