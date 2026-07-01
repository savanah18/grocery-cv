import requests
import numpy as np
import cv2
import aiohttp
import asyncio

import time

# Generate random image data
async def simulate_async(image_data):
    while True:
        # Send the data to the Flask server
        #print(f"Size of image data: {len(image_data)} bytes")
        start_time = time.time()
        url = 'http://127.0.0.1:5000/process_frame'
        # files = {'frame': ('frame.jpg', image_data, 'image/jpeg')}
        # async with aiohttp.ClientSession() as session:
        #     async with session.post(url, data=files) as response:
        #         if response.status == 200:
        #             # with open('processed_frame.jpg', 'wb') as f:
        #             #     f.write(await response.read())
        #             print(type(await response.read()))
        #         else:
        #             print(f"Failed to process frame: {response}")

        form = aiohttp.FormData()
        form.add_field('frame', image_data, filename='frame.jpg', content_type='image/jpeg')
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=form) as response:
                if response.status == 200:
                    # with open('processed_frame.jpg', 'wb') as f:
                    #     f.write(await response.read())
                    # print(type(await response.read()))
                    end_time = time.time()
                    print("fps",1 / (end_time - start_time))
                else:
                    print(f"Failed to process frame: {response.status}")

def simulate_http(image_data):
    while True:
        print(f"Size of image data: {len(image_data)} bytes")
        url = 'http://localhost:5000/process_frame'
        files = {'frame': ('frame.jpg', image_data, 'image/jpeg')}
        response = requests.post(url, files=files)

        if response.status_code == 200:
            print(type(response.content))
                
        else:
            print(f"Failed to process frame: {response}")

height, width = 480, 640
random_image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
_, buffer = cv2.imencode('.jpg', random_image)
image_data = buffer.tobytes()
asyncio.run(simulate_async(image_data=image_data))
#simulate_http(image_data=image_data)