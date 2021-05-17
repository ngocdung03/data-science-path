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
- Adding query parameters: `http://api.open-notify.org/iss-pass.json?lat=40.71&lon=-74`
    - It's almost always better to set up the parameters as a dictionary, because the requests library we mentioned earlier takes care of certain issues, like properly formatting the query parameters.
    ```json
    # Set up the parameters we want to pass to the API.
    # This is the latitude and longitude of New York City.
    parameters = {"lat": 40.71, "lon": -74}
    # Make a get request with the parameters.
    response = requests.get("http://api.open-notify.org/    iss-pass.json", params=parameters)
    # Print the content of the response (the data the   server returned)
    print(response.content)
    # This gets the same data as the command above
    response = requests.get("http://api.open-notify.org/    iss-pass.json?lat=40.71&lon=-74")
    print(response.content)
    ```
    - JSON format: encodes data structures like lists and dictionaries as strings to ensure that machines can read them easily. JSON is the main format for sending and receiving data through APIs.
    