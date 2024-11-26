import requests
import json

def fetch_and_print_itunes_data():
    # Define the URL for iTunes API
    url = "https://itunes.apple.com/search?term=thor&media=movie"

    try:
        # Make the GET request
        response = requests.get(url=url)
        
        # Check the response status code
        print("Status Code:", response.status_code)
        
        if response.status_code == 200:
            # Parse the JSON response
            json_object = json.loads(response.text)
            
            # Check if results are present
            if 'results' in json_object and len(json_object['results']) > 0:
                print("Movies Found:")
                for result in json_object['results']:
                    # Extract name and duration if available
                    track_name = result.get('trackName', 'Unknown Name')
                    track_duration = result.get('trackTimeMillis', None)
                    
                    # Convert duration to minutes and seconds if available
                    if track_duration:
                        minutes, seconds = divmod(track_duration // 1000, 60)
                        duration = f"{minutes}m {seconds}s"
                    else:
                        duration = "Unknown Duration"
                    
                    # Print the name and duration
                    print(f"Name: {track_name}, Duration: {duration}")
            else:
                print("No movies found in the response.")
        else:
            print("Failed to fetch data. Status Code:", response.status_code)

    except requests.exceptions.RequestException as e:
        # Handle exceptions such as connection errors or timeouts
        print("An error occurred:", e)

# Run the function
fetch_and_print_itunes_data()
