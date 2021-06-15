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
        - json library in Python: 2 main methods: dumps - take a Python object and converts to a string; loads - take a JSON string and converts to a Python object.
        - We can convert lists and dictionaries to JSON, and vice versa
        ```py
        # Make a list of fast food chains.
    best_food_chains = ["Taco Bell", "Shake Shack",     "Chipotle"]
    print(type(best_food_chains))
    # Import the JSON library.
    import json
    # Use json.dumps to convert best_food_chains to a   string.
    best_food_chains_string = json.dumps(best_food_chains)
    print(type(best_food_chains_string))
    # Convert best_food_chains_string back to a list.
    print(type(json.loads(best_food_chains_string)))
    # Make a dictionary
    fast_food_franchise = {
        "Subway": 24722,
        "McDonalds": 14098,
        "Starbucks": 10821,
        "Pizza Hut": 7600
    }
    # We can also dump a dictionary to a string and load    it.
    fast_food_franchise_string = json.dumps (fast_food_franchise)
    print(type(fast_food_franchise_string))
    # Use the JSON function loads to convert    fast_food_franchise_string to a Python object.
    fast_food_franchise_2 = json.loads  (fast_food_franchise_string)
    ```
    - get the content of a response as a Python object by .json() method:
    ```py
    # Make the same request we did two screens ago.
    parameters = {"lat": 37.78, "lon": -122.41}
    response = requests.get("http://api.open-notify.org/    iss-pass.json", params=parameters)

    # Get the response data as a Python object.  Verify     that it's a dictionary.
    json_data = response.json()
    print(type(json_data))
    print(json_data)
    # Get the duration value of the ISS's first pass over   San Francisco 
    first_pass_duration = json_data["response"][0]  ["duration"]

- The server sends more than a status code and the data when it generates a response. It also sends metadata with information on how it generated the data and how to decode it. This information appears in the *response headers*: `responde.headers`
- OpenNotify has one more API endpoint, astros.json. It tells us how many people are currently in space.
```py
response = requests.get("http://api.open-notify.org/astros.json")
in_space_count = response.json()["number"]
```

##### Intermediate APIs
- To authenticate with API, we need to use an access token, which can have scopes and specific permissions
```json
# Create a dictionary of headers containing our Authorization header.
headers = {"Authorization": "token 1f36137fbbe1602f779300dad26e4c1b7fbab631"}
# Make a GET request to the GitHub API with our headers.
# This API endpoint will give us details about Vik Paruchuri.
response = requests.get("https://api.github.com/users/VikParuchuri", headers=headers)
# Print the content of the response.  As you can see, this token corresponds to the account of Vik Paruchuri.
print(response.json())
```
- Pagination: API provider will only return a certain number of records per page. You can specify the page number that you want to access. To access all of the pages, you need to write a loop.
- Since we've authenticated with our token, the system knows who we are, and it can show us some relevant information without us specifying our username: `user = requests.get("https://api.github.com/user", headers=headers).json()`
- POST requests: send information (instead of retrieve it), and to create objects on the API's server.
- Consult API's documentation to figure out which endpoints accept which types of requests: https://docs.github.com/en/rest
    - A successful POST request will usually return a 201 status code indicating that it created the object on the server. Sometimes, the API will return the JSON representation of the new object as the content of the response.
    ```json
    payload = {"name": "test"}
    requests.post("https://api.github.com/user/repos",  json=payload, headers = headers)
    ```
- Update an existing object, instead of creating a new one: 
    - PATCH requests: change a few attributes of an object, but don't want to resend the entire object to the server. 
    - PUT requests: send the complete object we're revising as a replacement for the server's existing version.
    ```json
    payload = {"description": "The best repository ever!", "name": "test"}
    response = requests.patch("https://api.github.com/repos/VikParuchuri/test", json=payload)
    ```
- DELETE request removes objects from the server: `response = requests.delete("https://api.github.com/repos/VikParuchuri/test")`
    - A successful DELETE request will usually return a 204 status code indicating that it successfully deleted the object.

##### Challenge: working with the reddit APIs
- OAuth: `{"Authorization": "bearer 13426216-4U1ckno9J5AiK72VRbpEeBaMSKk"}`
- Documentation for the /r/python/top endpoint: https://old.reddit.com/dev/api#GET_{sort}
```json
# Retrieve the /r/python subreddit's top posts for the past day.
response = requests.get("https://oauth.reddit.com/r/python/top", headers = {"Authorization": "bearer 13426216-4U1ckno9J5AiK72VRbpEeBaMSKk", "User-Agent": "Dataquest/1.0"}, params={"t":"day"})
python_top = response.json()
```
- Find the article with the most upvotes
```json
# The variable python_top is a dictionary containing information about all of the individual posts that reddit users submitted during the past day.
# More about python_top format: https://old.reddit.com/dev/api#listings
# Extract the list containing all of the posts:
python_top_articles = python_top['data']['children']
# Find the post with the most upvotes
most_upvoted = ''
most_upvotes = 0
for article in python_top_articles:
    ar = article['data']
    if ar['ups'] >= most_upvotes:
        most_upvoted = ar['id']
        most_upvotes = ar['ups']
``` 
- Get all of the comments on the /r/python subreddit's top post from the past day.
```json
# Now that you have the ID for the most upvoted post, you can retrieve the comments on it using the /r/{subreddit}/comments/{article} endpoint. Replace the items in brackets with the appropriate values: {subreddit} — The name of the subreddit the post appears in (note that we already included leading /r in the URL). Use python for the python subreddit, for example. {article} — The ID for the post with the comments we want to retrieve. It should look like this: 4b7w9u.
# Make a GET request to the URL -> get response data using the response's json method.
headers = {"Authorization": "bearer 13426216-4U1ckno9J5AiK72VRbpEeBaMSKk", "User-Agent": "Dataquest/1.0"}
comments = requests.get('https://oauth.reddit.com/r/python/comments/4b7w9u', headers = headers).json()
```

- Find the most upvoted top-level comment in comments.
```json
# The first item in the list contains information about the post, and the second item contains information about the comments.
# Reddit users can comment on comments. This means that comments have one more key than posts do. The additional key, replies, contains the nested comments. 
# It's easier to focus on top-level comments and ignore the nested replies.
comments
comments_list = comments[1]['data']['children']
most_upvoted_comment = ''
most_upvotes_comment = 0
for comment in comments_list:
    cm = comment['data']
    if cm['ups'] > most_upvotes_comment:
        most_upvoted_comment = cm['id']
        most_upvotes_comment = cm['ups']
```

- Make a POST request to the /api/vote endpoint to upvote the most upvoted comment from the last screen.
```json
headers = {"Authorization": "bearer 13426216-4U1ckno9J5AiK72VRbpEeBaMSKk", "User-Agent": "Dataquest/1.0"}
payload = {'dir': 1, 'id': most_upvoted_comment}
status = requests.post('https://oauth.reddit.com/api/vote', json = payload, headers=headers).status_code
```

##### Web Scraping