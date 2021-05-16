### Working with APIs
- An API is a collection of tools that allows different applications to interact.
- Organizations host their APIs on web servers.
- The API usually returns this data in JavaScript Object Notation (JSON) format. 
- An endpoint is a server route for retrieving specific data from an API
- *GET* request: retrieve data
```json
# Make a get request to get the latest position of the ISS from the OpenNotify API.
response = requests.get("http://api.open-notify.org/iss-now.json")
status_code = response.status_code
```
- Status codes relevant to GET requests:
    - 200 — Everything went okay, and the server returned a result (if any).
    - 301 — The server is redirecting you to a different endpoint. This can happen when a company switches domain names, or when an endpoint's name has changed.
    - 401 — The server thinks you're not authenticated. This happens when you don't send the right credentials to access an API (we'll talk about this in a later mission).
    - 400 — The server thinks you made a bad request. This can happen when you don't send the information the API requires to process your request (among other things).
    - 403 — The resource you're trying to access is forbidden, and you don't have the right permissions to see it.
    - 404 — The server didn't find the resource you tried to access.