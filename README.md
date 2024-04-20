# Simple transfer style bot
Simple telegram bot to transfer style from one picture to another
# Installation
Install docker from official site

Open a terminal/command linethe required directory where the repository will be located 

git clone https://github.com/overwhelmerrr/simple_transfer_style_bot

build docker image with "docker build -t YOUR_IMAGE_NAME ."

run docker container with "docker run -it --name YOUR_CONTAINER NAME --rm  -e API_TOKEN="YOUR_TOKEN" -p 4000:80 YOUR_IMAGE_NAME"

-----------------------------------------------------
done! now bot is running

# How to use

Start the bot by pressing the 'start' button. Than you can press Transfer style or enter command /transfer_style. Send both the content and style photos
<img width="1360" alt="screen_1" src="https://github.com/overwhelmerrr/simple_transfer_style_bot/assets/93338693/5c071dcb-85f6-404f-8d57-0f7b6599e41f">
<img width="1361" alt="screen_2" src="https://github.com/overwhelmerrr/simple_transfer_style_bot/assets/93338693/b486e9d5-4874-44c5-8b6a-157c0e67d0e9">

Also bot command /transfer_style has strict rules to end any state. For example If you've sent the wrong content image, you could reset with /transfer_style
<img width="1375" alt="screen_3" src="https://github.com/overwhelmerrr/simple_transfer_style_bot/assets/93338693/8e315e84-a9ac-4090-b1e9-3552dd288180">

# References 
https://pytorch.org/tutorials/advanced/neural_style_tutorial.html

# Contacts 
@zulose
